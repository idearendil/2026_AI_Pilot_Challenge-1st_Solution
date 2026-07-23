"""baseline BT(DLL) 의 rule XML 지정 — **다른 claude_code 모듈보다 먼저** import 할 것.

BT DLL 이 읽는 rule XML 경로는 환경변수 `AIP_RULE_XML` 로 전달하는데, 이 값은
`JSBSimAIPLib.dll` 이 로드되는 시점(= `claude_code.env_utils` → `DogFightEnvWrapper`
→ `JSBSimWrapper` import 체인, JSBSimWrapper.py 의 모듈 레벨 LoadLibrary)에 **딱 한 번**
읽혀 캐싱된다. 그 뒤에 `os.environ` 을 바꿔도 DLL 은 무시한다.

세팅이 늦으면 DLL 은 `./Rule.xml` → `./Rule_forTraining.xml` 로 폴백하는데 후자의 트리는
`Task_Empty` 라 **상대가 조종을 전혀 안 하고 직진만 한다**. 겉으로는 BT 상대와 싸우는
것처럼 보이지만 실제로는 표적기가 가만히 있는 것이라, BT 상대 승률이 거짓으로 100%
가까이 찍힌다. 그래서 이 모듈은 numpy/torch 조차 import 하지 않는 leaf 로 두고,
entrypoint(train.py, run_local_dogfight.py) 맨 위에서 먼저 호출한다.

Ray 병렬 학습에서는 worker 가 별도 프로세스라 driver 의 os.environ 이 자동으로 따라가지
않는다. `ray.init(runtime_env={"env_vars": ...})` 로 worker **프로세스 시작 시점에**
주입해야 한다(parallel.py 참고).
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_KEY = "AIP_RULE_XML"

# BT DLL 별 기본 rule XML.
BT_RULE_DEFAULTS = {
    "AIP_DCS_baseline.dll": "./Rule_BaselineCore.xml",
}
DEFAULT_BT_DLL = "AIP_DCS_baseline.dll"


def rule_for(dll_name: str = DEFAULT_BT_DLL, rule_xml: str = "") -> str:
    """해당 DLL 에 쓸 rule XML 경로(빈 문자열 = DLL 기본값에 맡김)."""
    return rule_xml or BT_RULE_DEFAULTS.get(Path(dll_name).name, "")


def apply_rule_env(dll_name: str = DEFAULT_BT_DLL, rule_xml: str = "") -> str:
    """AIP_RULE_XML 을 세팅한다. **반드시 claude_code.env_utils import 전에** 호출."""
    rule = rule_for(dll_name, rule_xml)
    if rule:
        os.environ[ENV_KEY] = rule
    return rule


def rule_at_import() -> str:
    """JSBSimAIPLib.dll 로드 시점에 실제로 캐싱된 AIP_RULE_XML 값."""
    from claude_code import env_utils   # 이미 import 돼 있어야 의미가 있다

    return getattr(env_utils, "RULE_XML_AT_IMPORT", "")


def check_rule_applied(dll_name: str = DEFAULT_BT_DLL, rule_xml: str = "") -> str:
    """원하는 rule 이 실제로 DLL 에 반영됐는지 검증(아니면 RuntimeError).

    조용히 Task_Empty BT(= 직진만 하는 표적)로 학습해서 가짜 승률을 보는 사고를 막는다.
    """
    want = rule_for(dll_name, rule_xml)
    got = rule_at_import()
    if want and got != want:
        raise RuntimeError(
            f"BT rule XML 이 DLL 에 반영되지 않았습니다: 필요={want!r} / "
            f"claude_code.env_utils import 시점 값={got or '(없음)'!r}.\n"
            "AIP_RULE_XML 은 JSBSimAIPLib.dll 로드(= env_utils import) 시점에 한 번만 "
            "캐싱되므로 그 전에 claude_code.bt_rule.apply_rule_env() 를 호출해야 합니다.\n"
            "이대로 두면 DLL 이 Rule_forTraining.xml(Task_Empty)로 폴백해 상대가 조종을 "
            "전혀 안 하고, BT 상대 승률이 거짓으로 100% 가까이 나옵니다."
        )
    return want


__all__ = ["ENV_KEY", "BT_RULE_DEFAULTS", "DEFAULT_BT_DLL", "rule_for",
           "apply_rule_env", "rule_at_import", "check_rule_applied"]
