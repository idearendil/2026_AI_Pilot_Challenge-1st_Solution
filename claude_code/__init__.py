"""claude_code: CPU 환경 독립형 PPO 학습 + 제출 패키지.

RLlib 의존성 없이 순수 PyTorch로 PPO를 직접 구현하며, DogFight 환경
(`DogFightWrapper`)을 그대로 사용한다. 학습 결과는 2-파일 번들
(`metadata.json` + `policy_weights.pkl.gz`)로 저장되고, `submission_client.py`가
UDP 클라이언트 경로로 대결 서버(BattleServer_V1.2_VeryLow 포함)에 연결한다.
"""
