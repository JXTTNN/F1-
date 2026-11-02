#!/usr/bin/env bash
# 把本次 CI 的诊断结论写进 ``ci-status`` 分支。
#
# 动机：本机到 GitHub API 常常不通/被限流，判断「云端到底哪个用例红了」
# 只能靠 git。与其等人在网页上翻日志，不如让云端自己把结论提交回仓库：
#
#     git fetch origin ci-status && git show origin/ci-status:ci-status/CI__test.txt
#
# 写入内容只取判定所需的最小集合：失败用例名、断言摘要、最慢用例、日志尾部。
# 写入失败绝不影响主流程（诊断是附加能力，不能反过来把构建搞红）。
#
# 用法：bash scripts/ci_diag.sh <workflow 名> <job 名>
set -u

WF="${1:-ci}"
JOB="${2:-job}"
SAFE_WF=$(printf '%s' "$WF" | tr ' ' '_')
SAFE_JOB=$(printf '%s' "$JOB" | tr ' ' '_')
DIR="ci-status"
OUT="$DIR/${SAFE_WF}__${SAFE_JOB}.txt"

mkdir -p "$DIR"

{
  echo "workflow : $WF"
  echo "job      : $JOB"
  echo "sha      : ${GITHUB_SHA:-unknown}"
  echo "ref      : ${GITHUB_REF_NAME:-unknown}"
  echo "event    : ${GITHUB_EVENT_NAME:-unknown}"
  echo "run      : ${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-unknown}/actions/runs/${GITHUB_RUN_ID:-0}"
  echo "time     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo

  shopt -s nullglob
  logs=(.ci/*.log)
  if [ ${#logs[@]} -eq 0 ]; then
    echo "(本 job 没有产出 .ci/*.log —— 失败发生在 pytest 之前的步骤，"
    echo " 请看上面 time 对应的 run 链接)"
  fi

  for f in "${logs[@]}"; do
    echo "================ $f ================"
    echo "---- 失败 / 错误用例 ----"
    grep -E "^(FAILED|ERROR) " "$f" | head -40 || true
    echo "---- 断言摘要 ----"
    grep -E "^E  " "$f" | head -40 || true
    echo "---- 最慢用例 ----"
    awk '/slowest .*durations/{p=1} p' "$f" | head -25 || true
    echo "---- 结果行 ----"
    grep -E "[0-9]+ (passed|failed)" "$f" | tail -3 || true
    echo "---- 超时 / 崩溃痕迹 ----"
    grep -E "Timeout|timed out|SIGKILL|Killed|MemoryError" "$f" | head -10 || true
    echo "---- 日志尾部 ----"
    tail -60 "$f" || true
    echo
  done
} > "$OUT"

git config user.name "ci-status-bot"
git config user.email "ci-status-bot@users.noreply.github.com"

# 从当前的 detached HEAD 拉出 ci-status 分支：只放诊断文件，不掺进任何源码。
git checkout -B ci-status >/dev/null 2>&1 || exit 0
git add -f "$DIR" >/dev/null 2>&1 || exit 0
git commit -m "ci-status: ${WF}/${JOB} @ ${GITHUB_SHA:0:7}" >/dev/null 2>&1 || exit 0

for _ in 1 2 3; do
  if git push -f origin ci-status >/dev/null 2>&1; then
    echo "诊断已写入 ci-status 分支：${OUT}"
    exit 0
  fi
  sleep 5
done

echo "::warning title=诊断写入失败::ci-status 分支推送失败（不影响构建结论）"
exit 0
