import glob
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List, Optional

import pandas as pd


_TIME_COL = "Time"
_CODE_COL = "code"
_ACTION_COL = "Action"
_GROUPED_COL = "ActionGrouped"


@dataclass(frozen=True)
class LogParseOptions:
    dedupe_granularity: str = "exact"  # exact|second|minute
    exclude_codes: Optional[Iterable[str]] = None


def _normalize_exclude_codes(exclude_codes: Optional[Iterable[str]]) -> List[str]:
    if not exclude_codes:
        return []
    result: List[str] = []
    for code in exclude_codes:
        if code is None:
            continue
        code_str = str(code).strip()
        if code_str:
            result.append(code_str)
    return result


def _apply_dedupe(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["Time_dt"] = pd.to_datetime(df[_TIME_COL], errors="coerce")
    df = df.sort_values(["Time_dt"], kind="mergesort")

    if granularity == "second":
        df["Time_dt"] = df["Time_dt"].dt.floor("s")
    elif granularity == "minute":
        df["Time_dt"] = df["Time_dt"].dt.floor("min")
    # granularity == "exact": no flooring

    df = df.drop_duplicates(subset=["Time_dt"], keep="first")
    return df


def parse_log_lines(lines: Iterable[str]) -> pd.DataFrame:
    data = []
    for line in lines:
        if "=" not in line:
            continue
        time_str, action_str = line.strip().split("=", 1)
        match = re.match(r"(\d+|:)", action_str.strip())
        code = match.group(1) if match else ""
        data.append((time_str.strip(), code, action_str.strip()))

    return pd.DataFrame(data, columns=[_TIME_COL, _CODE_COL, _ACTION_COL])


def load_and_clean_logs(
    path_glob: str,
    options: Optional[LogParseOptions] = None,
    filter_keywords: Optional[Iterable[str]] = None,
    rules_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load logs from a glob (e.g. r"\\\\server\\share\\2025-*-*log.txt").

    Returns DataFrame columns: Time, code, Action, Time_dt.
    """

    options = options or LogParseOptions()

    files = sorted(glob.glob(path_glob))
    if not files:
        return pd.DataFrame(columns=[_TIME_COL, _CODE_COL, _ACTION_COL, "Time_dt"])

    all_lines: List[str] = []
    for file_path in files:
        try:
            with open(file_path, encoding="utf-8") as f:
                all_lines.extend(f.readlines())
        except UnicodeDecodeError:
            # Some logs can be ANSI/GBK; fall back without crashing
            with open(file_path, encoding="gbk", errors="ignore") as f:
                all_lines.extend(f.readlines())

    df = parse_log_lines(all_lines)

    exclude_codes = _normalize_exclude_codes(options.exclude_codes)
    if exclude_codes:
        df = df[~df[_CODE_COL].isin(exclude_codes)]

    if rules_csv_path:
        _, filter_rules = load_action_rules_from_csv(rules_csv_path)
        df = apply_filter_rules(df, filter_rules)

    if filter_keywords:
        keywords = [str(k) for k in filter_keywords if str(k).strip()]
        if keywords:
            pattern = "|".join(re.escape(k) for k in keywords)
            df = df[~df[_ACTION_COL].str.contains(pattern, na=False)]

    df = _apply_dedupe(df, options.dedupe_granularity)
    return df


def collect_log_files_by_date(folder_path: str, start_date: date, end_date: date) -> List[str]:
    """Collect log files in a folder by filename date.

    Matches names like: YYYY-M-Dlog.txt or YYYY-MM-DDlog.txt.
    """

    if not folder_path:
        return []

    pattern = os.path.join(folder_path, "202*-*-*log.txt")
    files: List[str] = []

    for file_path in glob.glob(pattern):
        name = os.path.basename(file_path)
        m = re.search(r"^(\d{4})-(\d{1,2})-(\d{1,2})", name)
        if not m:
            continue

        try:
            y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            file_date = datetime(y, mth, d).date()
        except ValueError:
            continue

        if start_date <= file_date <= end_date:
            files.append(file_path)

    return sorted(files)


def load_and_clean_logs_from_files(
    file_paths: Iterable[str],
    options: Optional[LogParseOptions] = None,
    filter_keywords: Optional[Iterable[str]] = None,
    rules_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load logs from explicit file paths.

    Returns DataFrame columns: Time, code, Action, Time_dt.
    """

    options = options or LogParseOptions()
    file_list = [p for p in (file_paths or []) if p]
    if not file_list:
        return pd.DataFrame(columns=[_TIME_COL, _CODE_COL, _ACTION_COL, "Time_dt"])

    all_lines: List[str] = []
    for file_path in file_list:
        try:
            with open(file_path, encoding="utf-8") as f:
                all_lines.extend(f.readlines())
        except UnicodeDecodeError:
            with open(file_path, encoding="gbk", errors="ignore") as f:
                all_lines.extend(f.readlines())

    df = parse_log_lines(all_lines)

    exclude_codes = _normalize_exclude_codes(options.exclude_codes)
    if exclude_codes:
        df = df[~df[_CODE_COL].isin(exclude_codes)]

    if rules_csv_path:
        _, filter_rules = load_action_rules_from_csv(rules_csv_path)
        df = apply_filter_rules(df, filter_rules)

    if filter_keywords:
        keywords = [str(k) for k in filter_keywords if str(k).strip()]
        if keywords:
            pattern = "|".join(re.escape(k) for k in keywords)
            df = df[~df[_ACTION_COL].str.contains(pattern, na=False)]

    df = _apply_dedupe(df, options.dedupe_granularity)
    return df


def load_action_rules_from_csv(csv_path: str) -> tuple[List[dict], List[dict]]:
    """Load grouping and filtering rules from a single CSV.

    Group rules: rule_type=group, columns: group, pattern, optional priority, overwrite, enabled.
    Filter rules: rule_type=filter, columns: filter_pattern or pattern, exclude_code, optional priority, enabled.
    """

    if not csv_path or not os.path.exists(csv_path):
        return [], []

    try:
        rules_df = pd.read_csv(csv_path)
    except Exception:
        return [], []

    rules_df.columns = [c.strip() for c in rules_df.columns]
    if rules_df.empty:
        return [], []

    if "enabled" in rules_df.columns:
        rules_df = rules_df[rules_df["enabled"].fillna(1).astype(int) == 1]

    if rules_df.empty:
        return [], []

    if "priority" not in rules_df.columns:
        rules_df["priority"] = 1000
    else:
        rules_df["priority"] = pd.to_numeric(rules_df["priority"], errors="coerce").fillna(1000).astype(int)

    if "rule_type" not in rules_df.columns:
        rules_df["rule_type"] = "group"
    else:
        rules_df["rule_type"] = (
            rules_df["rule_type"].fillna("group").astype(str).str.strip().str.lower()
        )

    group_rules: List[dict] = []
    filter_rules: List[dict] = []

    group_df = rules_df[rules_df["rule_type"] == "group"].copy()
    if not group_df.empty and "group" in group_df.columns and "pattern" in group_df.columns:
        if "overwrite" not in group_df.columns:
            group_df["overwrite"] = 0
        else:
            group_df["overwrite"] = group_df["overwrite"].fillna(0).astype(int)

        group_df["group"] = group_df["group"].astype(str)
        group_df["pattern"] = group_df["pattern"].astype(str)
        group_df = group_df.sort_values(["priority", "group"])
        group_rules = group_df.to_dict(orient="records")

    filter_df = rules_df[rules_df["rule_type"] == "filter"].copy()
    if not filter_df.empty:
        pattern_col = "filter_pattern" if "filter_pattern" in filter_df.columns else "pattern"
        filter_df["filter_pattern"] = (
            filter_df[pattern_col].fillna("").astype(str) if pattern_col in filter_df.columns else ""
        )
        if "exclude_code" not in filter_df.columns:
            filter_df["exclude_code"] = ""
        else:
            filter_df["exclude_code"] = filter_df["exclude_code"].fillna("").astype(str)

        filter_df = filter_df[
            (filter_df["filter_pattern"].str.strip() != "")
            | (filter_df["exclude_code"].str.strip() != "")
        ]

        if not filter_df.empty:
            filter_df = filter_df.sort_values(["priority"])
            filter_rules = filter_df[["filter_pattern", "exclude_code", "priority"]].to_dict(
                orient="records"
            )

    return group_rules, filter_rules


def load_action_grouping_rules_from_csv(csv_path: str) -> List[dict]:
    """Load action grouping rules from a CSV.

    Required columns: group, pattern
    Optional columns: priority, overwrite, enabled, rule_type
    """

    group_rules, _ = load_action_rules_from_csv(csv_path)
    return group_rules


def _parse_csv_list(value: str) -> List[str]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


def apply_filter_rules(df: pd.DataFrame, filter_rules: Optional[List[dict]]) -> pd.DataFrame:
    if df is None or df.empty or not filter_rules:
        return df

    patterns: List[str] = []
    codes: List[str] = []

    for rule in filter_rules:
        pat = str(rule.get("filter_pattern", "")).strip()
        if pat:
            patterns.append(pat)

        code_str = str(rule.get("exclude_code", "")).strip()
        if code_str:
            codes.extend(_parse_csv_list(code_str))

    out = df
    if codes:
        out = out[~out[_CODE_COL].isin(_normalize_exclude_codes(codes))]

    if patterns:
        pattern = "|".join(f"(?:{p})" for p in patterns)
        try:
            out = out[~out[_ACTION_COL].str.contains(pattern, case=False, na=False, regex=True)]
        except Exception:
            return out

    return out


def apply_action_grouping(
    df: pd.DataFrame,
    rules: Optional[List[dict]] = None,
    rules_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Add/overwrite ActionGrouped based on loaded rules.

    First-match-wins by default; set overwrite=1 in a rule to override.
    """

    if df is None or df.empty:
        out = (df.copy() if df is not None else pd.DataFrame())
        if _GROUPED_COL not in out.columns and _ACTION_COL in out.columns:
            out[_GROUPED_COL] = out[_ACTION_COL]
        return out

    rules = rules if rules is not None else load_action_grouping_rules_from_csv(rules_csv_path or "")

    out = df.copy()
    out[_GROUPED_COL] = out[_ACTION_COL]

    if not rules:
        return out

    for rule in rules:
        pat = str(rule.get("pattern", "")).strip()
        group = str(rule.get("group", "")).strip()
        if not pat or not group:
            continue

        try:
            mask = out[_ACTION_COL].str.contains(pat, case=False, na=False, regex=True)
        except Exception:
            continue

        if int(rule.get("overwrite", 0)) == 1:
            target_mask = mask
        else:
            target_mask = mask & (out[_GROUPED_COL] == out[_ACTION_COL])

        out.loc[target_mask, _GROUPED_COL] = group

    return out


def load_filter_keywords_from_excel(excel_path: str, column: str = "Keyword") -> List[str]:
    if not excel_path or not os.path.exists(excel_path):
        return []
    df = pd.read_excel(excel_path, engine="openpyxl")
    if column not in df.columns:
        return []
    return df[column].dropna().astype(str).tolist()
