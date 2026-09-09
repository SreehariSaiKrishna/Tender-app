#!/usr/bin/env bash
# Deploy the Tender Intelligence Agent pipeline to AWS.
#
# Run this from an Ubuntu (WSL) terminal every time you want a code or
# template change to go live - it's the full sequence, in the right order,
# so a stale build (the exact mistake that broke the last two deploy
# attempts - `sam deploy` reads .aws-sam/build/template.yaml, not the repo
# root one, so editing template.yaml alone and skipping `sam build` silently
# redeploys the old version) can't happen again:
#
#   ./scripts/deploy.sh
#
# Override the alert email for this run with:
#   ALERT_EMAIL=someone@else.com ./scripts/deploy.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export AWS_PROFILE="${AWS_PROFILE:-tender-agent}"
ALERT_EMAIL="${ALERT_EMAIL:-sreeharisai@oaks.guru}"
REGION="ap-south-1"

echo "== Confirming AWS identity (AWS_PROFILE=${AWS_PROFILE}) =="
aws sts get-caller-identity --query 'Arn' --output text

echo
echo "== Building the container image =="
sam build

echo
echo "== Validating the template =="
sam validate --region "${REGION}"

echo
echo "== Deploying =="
sam deploy --parameter-overrides "AlertEmail=${ALERT_EMAIL}"

echo
echo "== Done =="
echo "If this is the first deploy (or AlertEmail changed), confirm the SNS"
echo "subscription link AWS just emailed to ${ALERT_EMAIL} - alarms can't"
echo "notify you until that's clicked."
