#!/bin/bash

# must match the hierarchy described in generate-idea skill

CONTAINER_WORK_DIR=/workspace
CONTAINER_UID=1000

CAMPAIGN_DIR=campaigns/letf

if [[ -z "$OPENROUTER_API_KEY" ]]; then
  echo "OPENROUTER_API_KEY environment variable is not set."
  exit 1
fi

if [[ ! -f "$CAMPAIGN_DIR/CAMPAIGN.md" ]]; then
  echo "idea.md not found in $CAMPAIGN_DIR. Please run the generate-idea skill first."
  exit 1
fi

docker \
  run \
  --rm \
  -e GOOSE_PROVIDER=openrouter \
  -e GOOSE_MODEL=deepseek/deepseek-v4-flash-0731 \
  -e GOOSE_SKILLS_DIR="${CONTAINER_WORK_DIR}/skils" \
  -e OPENROUTER_API_KEY \
  -v "./campaigns/letf:${CONTAINER_WORK_DIR}/work" \
  -v "./CONTEXT.md:${CONTAINER_WORK_DIR}/CONTEXT.md:ro" \
  -v "./summaries/letf:${CONTAINER_WORK_DIR}/refs:ro" \
  -v "./skills:${CONTAINER_WORK_DIR}/skills:ro" \
  aq-sandbox \
  "RUST_LOG=goose=debug,goose_mcp=trace NO_COLOR=1 goose run -t \"Generate a new trading idea. Use work as the directtory for your output. References are in refs/.\" | tee \"${CONTAINER_WORK_DIR}/work/run-$(date +%F_%H-%M-%S).log\""
