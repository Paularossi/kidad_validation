"""Append AI annotations to the dieticians' screenshot workbook (no agreement metrics).

For every screenshot, do the following:
1. Read participant ID and screenshot ID
2. Find the clip containing that screenshot
3. Check whether the screenshot is part of an AI-detected ad interval (Phase 1)
4. If it does, find the corresponding Phase 2 result
5. Write AI fields such as: clip ID, ad detected, ad group ID, relevance, etc.

"""
import argparse
import ast
from collections import Counter, defaultdict
from pathlib import Path
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['overall_category', 'product_categories', 'brand_business_categories',
          'brand_main', 'brand_other', 'type_of_marketing', 'marketing_strategies',
          'target_group', 'who_category', 'nova_category', 'confidence']
HEADERS = ['AI clip ID', 'AI ad detected', 'AI ad group ID', 'AI relevance',
           'AI processing status', 'AI round1 confidence'] + ['AI ' + field for field in FIELDS]


def identifier(value):
    text = str(value).strip()
    if re.fullmatch(r'\d+\.0', text):
        text = text[:-2]
    return text


def screenshot_id(path):
    return identifier(Path(str(path).replace('\\', '/')).stem)


def paths(value):
    result = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(result, (list, tuple)) or not result:
        raise ValueError('Missing or invalid ad_screenshot_paths')
    return list(result)


def read_table(path):
    return pd.read_excel(path, dtype={'enrol_number': str}, keep_default_na=False)


def unique_records(data, key):
    result = {}
    for row in data.to_dict('records'):
        k = key(row)
        if k in result:
            raise ValueError(f'Duplicate input record: {k}')
        result[k] = row
    return result


def build_lookup(aggregated, validation):
    clips = unique_records(read_table(aggregated / 'all_results.xlsx'),
                           lambda r: (identifier(r['enrol_number']), str(r['id'])))
    results = unique_records(read_table(aggregated / 'round2_results.xlsx'), lambda r: str(r['ad_id']))
    ads = read_table(aggregated / 'all_ads_round2.xlsx')
    coverage = defaultdict(set)
    for path in sorted(validation.glob('*/frames.xlsx')):
        for row in read_table(path).to_dict('records'):
            enrol = identifier(row.get('enrol_number', path.parent.name))
            coverage[(enrol, screenshot_id(row['filename']))].add((enrol, str(row['clip_id'])))
    groups = defaultdict(set)
    by_id = {}
    clip_ads = Counter()
    mismatches = set()
    for row in ads.to_dict('records'):
        enrol, clip = identifier(row['enrol_number']), str(row['id'])
        group = f"{enrol}_{clip}_f{int(row['start_frame'])}"
        if group in by_id:
            raise ValueError(f'Duplicate ad group: {group}')
        by_id[group] = row
        clip_ads[(enrol, clip)] += 1
        screenshots = [screenshot_id(p) for p in paths(row['ad_screenshot_paths'])]
        result = results.get(group)
        if result and ('ad_screenshot_paths' not in result or
                       [screenshot_id(p) for p in paths(result['ad_screenshot_paths'])] != screenshots):
            mismatches.add(group)
        for shot in screenshots:
            key = (enrol, shot)
            if (enrol, clip) not in coverage.get(key, set()):
                raise ValueError(f'Ad screenshot absent from clip frame metadata: {group}, {shot}')
            groups[key].add(group)
    return clips, results, coverage, groups, clip_ads, mismatches


def annotation(key, lookup):
    clips, results, coverage, groups, clip_ads, mismatches = lookup
    out = dict.fromkeys(HEADERS, '')
    matched_clips = coverage.get(key, set())
    matched_groups = sorted(groups.get(key, set()))
    out['AI clip ID'] = '; '.join(sorted(c[1] for c in matched_clips))
    out['AI ad group ID'] = '; '.join(matched_groups)
    if len(matched_clips) > 1 or len(matched_groups) > 1:
        out['AI processing status'] = 'AMBIGUOUS_MATCH'
        return out
    if not matched_clips:
        out['AI processing status'] = 'SCREENSHOT_NOT_FOUND'
        return out
    clip_key = next(iter(matched_clips))
    clip = clips.get(clip_key)
    if not clip:
        out['AI processing status'] = 'ROUND1_MISSING'
        return out
    label = clip.get('label')
    # round 1 confidence (clip-wise)
    if not clip.get('error') and label in {'AD', 'NON_AD', 'UNCLEAR'}:
        out['AI round1 clip confidence'] = clip.get('confidence', '')
    if clip.get('error') or label not in {'AD', 'NON_AD'}:
        out['AI ad detected'] = 'UNCLEAR' if label == 'UNCLEAR' else ''
        out['AI processing status'] = 'ROUND1_UNCLEAR' if label == 'UNCLEAR' else 'ROUND1_FAILED'
        return out
    if matched_groups and label != 'AD':
        out['AI processing status'] = 'INCONSISTENT_ROUND1'
        return out
    if not matched_groups:
        if label == 'AD':
            try:
                complete = int(clip['n_ads']) == clip_ads[clip_key] and clip_ads[clip_key] > 0
            except (ValueError, KeyError, TypeError):
                complete = False
            if not complete:
                out['AI processing status'] = 'AD_MAPPING_INCOMPLETE'
                return out
        out['AI ad detected'] = 'NO'
        out['AI processing status'] = 'OK'
        return out
    out['AI ad detected'] = 'YES'
    group = matched_groups[0]
    result = results.get(group)
    if group in mismatches:
        out['AI processing status'] = 'ROUND2_INPUT_MISMATCH'
    elif not result:
        out['AI processing status'] = 'ROUND2_MISSING'
    elif result.get('error'):
        out['AI processing status'] = 'ROUND2_FAILED'
    elif result.get('ad_relevance') not in {'YES', 'NO', 'UNSURE'}:
        out['AI processing status'] = 'ROUND2_INVALID'
    else:
        out['AI processing status'] = 'OK'
        out['AI relevance'] = result['ad_relevance']
        for field in FIELDS:
            out['AI ' + field] = result.get(field, '')
    return out


def append_columns(source, output, sheet_name, header_row, start_column, records):
    """Patch only the sheet XML; preserve original cells, validations and other ZIP parts."""
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    main = ns['m']
    with zipfile.ZipFile(source) as archive:
        book = ET.fromstring(archive.read('xl/workbook.xml'))
        sheet = next(s for s in book.find('m:sheets', ns) if s.get('name') == sheet_name)
        rid = sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        rels = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        target = next(r.get('Target') for r in rels if r.get('Id') == rid)
        part = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl', target))
        xml = archive.read(part).decode('utf-8')
        # Keep the existing XML verbatim, including extension namespaces/validations.
        def patch_row(match):
            row = int(match.group(2))
            values = HEADERS if row == header_row else records.get(row)
            if values is None:
                return match.group(0)
            cells = []
            for col, value in enumerate(values, start_column):
                cell = ET.Element('c', {'r': f'{get_column_letter(col)}{row}', 't': 'inlineStr'})
                text = ET.SubElement(ET.SubElement(cell, 'is'), 't')
                text.text = str(value)
                cells.append(ET.tostring(cell, encoding='unicode'))
            return match.group(1) + match.group(3) + ''.join(cells) + '</row>'
        xml = re.sub(r'(<row\b[^>]*\br="(\d+)"[^>]*>)(.*?)</row>', patch_row, xml, flags=re.S)
        last_col = get_column_letter(start_column + len(HEADERS) - 1)
        max_row = max(records, default=header_row)
        xml = re.sub(r'<dimension\b[^>]*/>', f'<dimension ref="A1:{last_col}{max_row}"/>', xml, count=1)
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as dest:
            for info in archive.infolist():
                dest.writestr(info, xml.encode('utf-8') if info.filename == part else archive.read(info.filename))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codebook', type=Path, default=ROOT / 'data/dieticians/Codebook_testing_KidAd study_V2_NL.xlsx')
    parser.add_argument('--aggregated', type=Path, default=ROOT / 'data/aggregated validation')
    parser.add_argument('--validation', type=Path, default=ROOT / 'data/validation 1')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/aggregated validation/dieticians_with_ai.xlsx')
    parser.add_argument('--sheet', default='Coding')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.output.resolve() == args.codebook.resolve():
        parser.error('Output must differ from the original codebook')
    lookup = build_lookup(args.aggregated, args.validation)
    book = load_workbook(args.codebook, read_only=True, data_only=False)
    sheet = book[args.sheet]
    header_row = None
    for row in sheet.iter_rows(min_row=1, max_row=10):
        values = [str(c.value).strip() for c in row]
        if 'User' in values and 'Id_number' in values:
            header_row = row[0].row
            enrol_col, shot_col = values.index('User'), values.index('Id_number')
            break
    if header_row is None:
        raise ValueError('Cannot locate User and Id_number headers')
    records, counts = {}, Counter()
    for row in sheet.iter_rows(min_row=header_row + 1):
        enrol, shot = identifier(row[enrol_col].value), identifier(row[shot_col].value)
        if not enrol.isdigit() or not shot.isdigit():
            continue  # Preserve blank rows and examples without adding AI labels.
        result = annotation((enrol, shot), lookup)
        records[row[0].row] = [result[h] for h in HEADERS]
        counts[result['AI processing status']] += 1
    start_column = sheet.max_column + 1
    book.close()
    print(f'Mapped {len(records)} screenshot rows: {dict(counts)}')
    if not args.dry_run:
        if args.output.exists():
            raise FileExistsError(f'Choose a new --output path; refusing to overwrite {args.output}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        append_columns(args.codebook, args.output, args.sheet, header_row, start_column, records)
        print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
