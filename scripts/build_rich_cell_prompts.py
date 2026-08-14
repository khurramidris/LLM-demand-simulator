from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_SCALE = 590.0
HISTORY_START = pd.Timestamp("2018-09-01")
HISTORY_END = pd.Timestamp("2019-09-19")


def read_single_csv_from_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX/') and '/._' not in n]
        if len(members) != 1:
            raise RuntimeError(f"Expected one real CSV in {path}, found {members}")
        with zf.open(members[0]) as handle:
            return pd.read_csv(handle)


def qcut(values: pd.Series, labels: list[str]) -> pd.Series:
    try:
        out = pd.qcut(values, q=len(labels), labels=labels, duplicates='drop')
        if out.nunique(dropna=True) >= 2:
            return out.astype('object')
    except ValueError:
        pass
    ranked = values.rank(method='average')
    n_bins = min(len(labels), max(int(ranked.nunique()), 1))
    if n_bins <= 1:
        return pd.Series([labels[0]] * len(values), index=values.index, dtype='object')
    return pd.qcut(ranked, q=n_bins, labels=labels[:n_bins], duplicates='drop').astype('object')


def assign_original_cells(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    df['customer_id'] = df['customer_id'].astype(str)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['txn_count'] = pd.to_numeric(df['txn_count'], errors='coerce').fillna(0.0)
    df['mean_price'] = pd.to_numeric(df['mean_price'], errors='coerce').fillna(0.0)
    df['top_product_type'] = df['top_product_type'].fillna('UNKNOWN').astype(str)
    df = df[df['txn_count'] > 0].copy()

    span_months = ((HISTORY_END - HISTORY_START).days + 1) / 30.4375
    df['txn_per_month'] = df['txn_count'] / span_months
    df['age_bin'] = pd.cut(
        df['age'], bins=[16,25,35,45,55,200], right=False,
        labels=['16-24','25-34','35-44','45-54','55+']
    ).astype('object')
    df['engagement_bin'] = qcut(df['txn_per_month'], ['low','mid','high'])
    df['price_tier'] = qcut(df['mean_price'], ['low','mid','high'])
    top_types = df['top_product_type'].value_counts().head(100).index
    df['taste_bucket'] = np.where(df['top_product_type'].isin(top_types), df['top_product_type'], 'OTHER')
    df['age_bin'] = df['age_bin'].fillna('UNKNOWN')
    return df


def build_trouser_history(path: Path) -> pd.DataFrame:
    tx = read_single_csv_from_zip(path)
    tx['t_dat'] = pd.to_datetime(tx['t_dat'], errors='coerce')
    tx = tx[(tx['t_dat'] >= HISTORY_START) & (tx['t_dat'] <= HISTORY_END)].copy()
    tx['customer_id'] = tx['customer_id'].astype(str)
    tx['article_id'] = pd.to_numeric(tx['article_id'], errors='coerce')
    tx['price'] = pd.to_numeric(tx['price'], errors='coerce') * PRICE_SCALE
    tx = tx.dropna(subset=['customer_id','article_id','price','t_dat'])
    tx = tx.sort_values(['customer_id','t_dat'])

    g = tx.groupby('customer_id', sort=False)
    out = g.agg(
        tr_txn_count=('article_id','size'),
        tr_unique_articles=('article_id','nunique'),
        tr_mean_price=('price','mean'),
        tr_last_date=('t_dat','max'),
        tr_active_days=('t_dat','nunique'),
    ).reset_index()
    out['tr_days_since_last'] = (HISTORY_END - out['tr_last_date']).dt.days.astype(float)
    out['tr_repeat_ratio'] = 1.0 - out['tr_unique_articles'] / out['tr_txn_count'].clip(lower=1)
    for days in [30,90,180]:
        start = HISTORY_END - pd.Timedelta(days=days-1)
        counts = tx.loc[tx['t_dat'] >= start].groupby('customer_id').size().rename(f'tr_txn_last_{days}d')
        out = out.merge(counts, on='customer_id', how='left')
    out = out.drop(columns=['tr_last_date'])
    return out


def fmt_range(lo: float, hi: float, digits: int = 1) -> str:
    return f"{lo:.{digits}f}–{hi:.{digits}f}"


def main() -> None:
    root = Path('outputs/research/rich_persona_signal')
    root.mkdir(parents=True, exist_ok=True)
    features = read_single_csv_from_zip(Path('outputs/personas/customer_features.csv.zip'))
    cells = pd.read_csv('outputs/personas/persona_cells.csv').head(50).copy()
    members = assign_original_cells(features)
    tr = build_trouser_history(Path('outputs/products/txns_trousers_online.csv.zip'))
    members = members.merge(tr, on='customer_id', how='left')

    key = ['age_bin','engagement_bin','price_tier','taste_bucket']
    matched = members.merge(cells[key + ['persona_id','n_customers','persona_prompt']], on=key, how='inner')
    rows = []
    numeric = ['tr_txn_count','tr_unique_articles','tr_mean_price','tr_days_since_last','tr_repeat_ratio','tr_txn_last_30d','tr_txn_last_90d','tr_txn_last_180d']
    for persona_id, group in matched.groupby('persona_id', sort=False):
        base = group['persona_prompt'].iloc[0].split('\n\nTask:')[0].strip()
        has_tr = group['tr_txn_count'].notna()
        share = float(has_tr.mean())
        buyers = group.loc[has_tr].copy()
        if len(buyers):
            stats = {c: (float(buyers[c].quantile(.25)), float(buyers[c].quantile(.75))) for c in numeric}
            extra = (
                f" Historical behavioral context for this same customer segment, using only purchases before {HISTORY_END.date()}: "
                f"about {share*100:.1f}% bought trousers at least once. Among those trouser buyers, the middle 50% made "
                f"{fmt_range(*stats['tr_txn_count'], 0)} trouser purchases across {fmt_range(*stats['tr_unique_articles'],0)} distinct trouser articles, "
                f"paid about ${fmt_range(*stats['tr_mean_price'],2)}, and last bought trousers {fmt_range(*stats['tr_days_since_last'],0)} days before the cutoff. "
                f"Their repeat-item tendency was {fmt_range(*stats['tr_repeat_ratio'],2)}; middle-50% trouser counts in the last 30/90/180 days were "
                f"{fmt_range(*stats['tr_txn_last_30d'],0)}, {fmt_range(*stats['tr_txn_last_90d'],0)}, and {fmt_range(*stats['tr_txn_last_180d'],0)}."
            )
        else:
            extra = (
                f" Historical behavioral context for this same customer segment, using only purchases before {HISTORY_END.date()}: "
                "no members in the available history bought trousers."
            )
        instruction = (
            '\n\nTask: Given a product and a list of prices, return the probability you would buy at each price. '
            'Output JSON exactly: {"prices": [...], "p_buy": [...], "reason": "<=30 words"}.'
        )
        rows.append({
            'persona_id': persona_id,
            'n_customers': int(group['n_customers'].iloc[0]),
            'base_persona_prompt': group['persona_prompt'].iloc[0],
            'rich_persona_prompt': base + extra + instruction,
            'n_matched_customers': int(len(group)),
            'share_with_trouser_history': share,
        })

    out = pd.DataFrame(rows).sort_values('persona_id')
    out.to_csv(root / 'matched_rich_persona_prompts.csv', index=False)
    manifest = {
        'protocol': 'MATCHED-CELL-RICH-PROMPTS-v1',
        'history_start': str(HISTORY_START.date()),
        'history_end': str(HISTORY_END.date()),
        'n_personas': int(len(out)),
        'principle': 'Same original population cells and memberships; treatment only enriches prompt with pre-cutoff trouser behavioral summaries.',
        'columns': list(out.columns),
    }
    (root / 'matched_rich_persona_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(out[['persona_id','n_customers','n_matched_customers','share_with_trouser_history']].to_string(index=False))


if __name__ == '__main__':
    main()
