# -*- coding: utf-8 -*-
"""f16.xml <aerodynamics> 자동추출 → cuda_fdm/ref/f16_aero_data.py.

각 <axis>의 <function>을 파싱: 이름/축/상수인자(property)/스칼라값/table(1D·2D).
JSBSim 공력함수는 거의 균일하게 product(qbar, Sw, [factors...], table|value) 구조.
kCLge 같은 최상위 함수(축 밖, product 없이 table 직결)도 별도 수집.

산출물은 Python literal (double 그대로). CUDA 포팅시 이 데이터로 C 헤더도 생성 가능.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

XML = Path(r"D:\AIP_LIB_claude\DogFightEnv\Release\aircraft\f16\f16.xml")
OUT = Path(__file__).resolve().parents[1] / "ref" / "f16_aero_data.py"


def parse_table(tel):
    """<table> → dict. 1D: {'type':'1d','indep':name,'rows':[[x,y]...]}
       2D: {'type':'2d','row_indep','col_indep','rowvals','colvals','data'}"""
    indeps = tel.findall("independentVar")
    data_txt = tel.find("tableData").text
    if len(indeps) <= 1:
        indep = indeps[0].text.strip() if indeps else None
        rows = []
        for line in data_txt.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                rows.append([float(parts[0]), float(parts[1])])
        return {"type": "1d", "indep": indep, "rows": rows}
    else:
        row_indep = col_indep = None
        for iv in indeps:
            lu = iv.get("lookup")
            if lu == "row":
                row_indep = iv.text.strip()
            elif lu == "column":
                col_indep = iv.text.strip()
        lines = [ln for ln in data_txt.strip().splitlines() if ln.split()]
        colvals = [float(x) for x in lines[0].split()]
        rowvals = []
        data = []
        for ln in lines[1:]:
            p = ln.split()
            rowvals.append(float(p[0]))
            data.append([float(x) for x in p[1:]])
        return {"type": "2d", "row_indep": row_indep, "col_indep": col_indep,
                "rowvals": rowvals, "colvals": colvals, "data": data}


def parse_function(fel):
    """<function> → dict(name, factors:[property...], value:float|None, table:dict|None)."""
    name = fel.get("name")
    prod = fel.find("product")
    container = prod if prod is not None else fel
    factors = []
    value = None
    table = None
    for child in container:
        tag = child.tag
        if tag == "property":
            factors.append(child.text.strip())
        elif tag == "value":
            value = float(child.text.strip())
        elif tag == "table":
            table = parse_table(child)
        elif tag == "description":
            pass
    return {"name": name, "factors": factors, "value": value, "table": table}


def main():
    root = ET.parse(XML).getroot()
    aero = root.find("aerodynamics")
    top_functions = {}   # 축 밖 함수 (kCLge)
    axes = {}
    for el in aero:
        if el.tag == "function":
            f = parse_function(el)
            top_functions[f["name"]] = f
        elif el.tag == "axis":
            axname = el.get("name")
            axes[axname] = [parse_function(f) for f in el.findall("function")]

    lines = ["# -*- coding: utf-8 -*-",
             '"""자동생성: f16.xml aerodynamics 추출 (extract_f16.py). 수정하지 말 것."""',
             "TOP_FUNCTIONS = " + repr(top_functions),
             "AXES = " + repr(axes),
             ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    # 요약
    print(f"wrote {OUT}")
    print("top functions:", list(top_functions))
    for ax, fs in axes.items():
        print(f"  axis {ax}: {len(fs)} functions")
        for f in fs:
            t = f["table"]["type"] if f["table"] else ("value" if f["value"] is not None else "?")
            print(f"    {f['name']:30s} factors={f['factors']} val={f['value']} table={t}")


if __name__ == "__main__":
    main()
