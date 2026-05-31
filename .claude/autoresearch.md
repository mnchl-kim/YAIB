# autoresearch program

> 본 문서는 autoresearch 에이전트(Claude/Codex 등)의 동작 가이드다.
> 사람은 이 파일을 편집해서 에이전트의 행동을 조정한다.
> 에이전트는 이 파일을 직접 수정하지 않는다.

## 미션

`TASK.local.md` plan 1. Implement all baselines and validate on `hirid / mortality`.
실험 진행해줘. **Latest SOTA는 진행 금지.**

대상 baseline (총 5개 실험험):
- GRU-D
- BRITS
- SeFT
- mTAND
- LatentODE

각 baseline 은 별도 worktree/branch에서 구현하고 `hirid / mortality24` 에서 smoke test 까지 완료한다.

## 베이스라인

- 실행 entry point:
  ```bash
  /team/team_bs_ic/personal/mincheol.kim/git/YAIB/paper/scripts/run.sh \
      -d hirid -t mortality24 -m <model> -g <gpu> -j <threads>
  ```
- 실험 결과는 다음 경로에 자동 수집된다:
  ```
  /team/team_bs_ic/personal/mincheol.kim/git/YAIB/logs/<dataset>/<task>/<model>/<timestamp>/
  ```
- 이미 저장된 GRU / LGBMClassifier 의 결과 포맷을 참고한다.

## 평가 메트릭

학습 종료 후 다음 파일이 생성되어야 한다:

```
/team/team_bs_ic/personal/mincheol.kim/git/YAIB/logs/<dataset>/<task>/<model>/<timestamp>/accumulated_test_metrics.json
```

내용 예 (GRU baseline):
```json
{"avg": {"loss": ..., "AUC": 0.8417, "PR": 0.3748},
 "std": {"loss": ..., "AUC": 0.0031, "PR": 0.0058},
 "CI_0.95": {...}, "execution_time": ...}
```

**성공 기준**: `AUC` mean±std, `PR` mean±std 값이 정상적으로 기록되어 있으면 해당 실험은 성공으로 평가한다. 절대 수치(다른 baseline 대비 성능)는 이 단계에서 평가 대상이 아니다 (smoke test).

파싱 예:
```bash
cat /team/team_bs_ic/personal/mincheol.kim/git/YAIB/logs/hirid/mortality24/<MODEL>/<ts>/accumulated_test_metrics.json | jq '{AUC_mean: .avg.AUC, AUC_std: .std.AUC, PR_mean: .avg.PR, PR_std: .std.PR}'
```

## 수정 가능한 파일

- 새 baseline 모델 구현 파일 (`icu_benchmarks/models/...` 또는 worktree 내 신규 파일)
- 해당 baseline 을 등록하기 위한 최소한의 config 추가
- 실험 pipeline 은 **최소한의 변경만 적용** (다른 baseline과 공정한 실험결과 비교 가능해야 함)

## 수정 불가능한 파일

- `CLAUDE.md`
- `.claude/TASK.md`
- `.claude/autoresearch.md` (이 문서)
- 로그 `/team/team_bs_ic/personal/mincheol.kim/git/YAIB/logs/*`
- 데이터 `/team/team_bs_ic/personal/mincheol.kim/git/YAIB-cohorts/data*`
- 다른 worktree/branch 의 파일 일체

## 실험 1개 워크플로우

1. **실험 방향 수립**: 어떤 baseline 을 어떤 reference 구현을 따라, 어떤 최소 인터페이스로 YAIB pipeline 에 통합할지 한 줄로 적는다. (예: "GRU-D, https://github.com/PeterChe1990/GRU-D 의 cell 구현을 YAIB DLPredictionWrapper 에 wrap")
2. **worktree 생성**: baseline 당 worktree/branch 1개. `TASK.local.md` 의 `### Baselines` 항목에 paper/code URL 을 먼저 기록.
3. **구현**: 위 "수정 가능한 파일" 범위 안에서만.
4. **실행**: `scripts/run.sh -d hirid -t mortality24 -m <model> -g <gpu> -j <threads>`.
   - GPU `0-5` 만 사용, 1 experiment / GPU, `-j 16` 권장, 동시 max 6.
   - `screen` 안에서 실행 (1 window = 1 agent = 1 worktree/branch/GPU).
5. **결과 파싱**: `accumulated_test_metrics.json` 에서 `avg.AUC`, `std.AUC`, `avg.PR`, `std.PR` 추출.
6. **로그 누적**: `/team/team_bs_ic/personal/mincheol.kim/git/YAIB/paper/.claude/autoresearch.log` 에 한 줄 append:
   ```
   [YYYY-MM-DD HH:MM:SS] 실험 X done on GPU Y | model=<MODEL> | dataset=hirid | task=mortality24 | AUC=<mean>±<std> | PR=<mean>±<std> | time=HH:MM:SS | hyp="<가설>" | change="<diff 요약>"
   ```
7. **판단 & 커밋**:
   - **성공 (metric 정상 기록)**:
     `git add -A && git commit -m "trial X: <MODEL> baseline on hirid/mortality24, AUC=<mean>±<std>"`
   - **실패 (학습 crash / metric 누락)**:
     실패 원인을 autoresearch.log 에 같이 기록하고, 코드 수정 후 성공할 때까지 반복복.

## 금지 사항

- 수정 불가능한 파일 변경 시도 금지
- 새로 생성한 worktree 이외의 모든 파일 일체 수정 금지
- YAIB protocol 변경 시도 금지
- 동시 실험 진행 시 같은 GPU 중복 점유 금지
- Latest SOTA baseline 진행 금지

## 시작 명령

사람이 다음과 같이 시작 신호를 준다:

> "autoresearch.md 봤지? 실험 시작하자"

이때 에이전트는:
1. 새로운 worktree 생성 (baseline 1개당 1 worktree)
2. 사용할 GPU, CPU thread resource 확인 (`nvidia-smi`, 다른 screen window 점유 확인)
3. baseline 결정 → official paper/code 를 `TASK.local.md` 에 기록 → 구현 → 실행 → 결과 기록
4. 평가 메트릭 달성할 때까지 반복
