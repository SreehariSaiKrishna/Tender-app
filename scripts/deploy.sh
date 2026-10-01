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
echo "== Reading the dashboard URL from the frontend stack =="
# The API's CORS origin and Cognito's login redirect are both this URL.
FRONTEND_URL=$(aws cloudformation describe-stacks --stack-name tender-agent-frontend --region "${REGION}"   --query "Stacks[0].Outputs[?OutputKey=='FrontendUrl'].OutputValue" --output text)
if [ -z "$FRONTEND_URL" ] || [ "$FRONTEND_URL" = "None" ]; then
  echo "Could not read the frontend stack's FrontendUrl output - deploy it first (./scripts/deploy-frontend.sh)."
  exit 1
fi
echo "Frontend URL: ${FRONTEND_URL}"

echo
echo "== Building the container image =="
sam build

echo
echo "== Validating the template =="
sam validate --region "${REGION}"

echo
echo "== Deploying =="
sam deploy --parameter-overrides "AlertEmail=${ALERT_EMAIL}" "FrontendUrl=${FRONTEND_URL}"

echo
echo "== Branding the Cognito login page =="
# OAKS logo + colours on the hosted page the dashboard's "Sign in" button
# opens (see branding/). Not in the template: CloudFormation can set the
# page's CSS but not its logo, and one call sets both together.
POOL_ID=$(aws cloudformation describe-stacks --stack-name tender-agent --region "${REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='UserPoolId'].OutputValue" --output text)
aws cognito-idp set-ui-customization --region "${REGION}" --user-pool-id "${POOL_ID}" --client-id ALL \
  --css "$(cat branding/cognito-login.css)" --image-file fileb://branding/oaks-wordmark.png > /dev/null
echo "Login page branded."

echo
echo "== Done =="
echo "If this is the first deploy (or AlertEmail changed), confirm the SNS"
echo "subscription link AWS just emailed to ${ALERT_EMAIL} - alarms can't"
echo "notify you until that's clicked."
