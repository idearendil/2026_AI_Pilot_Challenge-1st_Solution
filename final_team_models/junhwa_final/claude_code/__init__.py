"""claude_code: 독립형 PPO 학습 + 제출 패키지.

원본 프레임워크(train_rllib.py / RLlib)와 *완전히 동일한* DogFight 환경
(`DogFightWrapper`)을 그대로 사용하되, RLlib 의존성 없이 순수 PyTorch로
PPO를 직접 구현한다. 학습 결과는 원본과 같은 2-파일 번들
(`metadata.json` + `policy_weights.pkl.gz`)로 저장되며, `submission.py`가
원본 `student/my_submission.py`와 동일한 UDP 클라이언트 경로로 대결 서버에
연결한다.
"""
