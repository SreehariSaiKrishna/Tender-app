#!/usr/bin/env bash
# Deploy the Tender Intelligence Agent frontend (S3 + CloudFront).
#
# Deliberately a separate script from scripts/deploy.sh (the backend) - the
# two stacks are unrelated infrastructure. The only place they connect is
# here: this script reads the *backend* stack's ApiUrl output and bakes it
# into index.html as a plain static string before uploading - not a
# CloudFormation cross-stack reference, just a deploy-time convenience.
#
#   ./scripts/deploy-frontend.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export AWS_PROFILE="${AWS_PROFILE:-tender-agent}"
REGION="ap-south-1"
BACKEND_STACK="tender-agent"
FRONTEND_STACK="tender-agent-frontend"

echo "== Confirming AWS identity (AWS_PROFILE=${AWS_PROFILE}) =="
aws sts get-caller-identity --query 'Arn' --output text

echo
echo "== Deploying frontend infrastructure (S3 + CloudFront) =="
(cd frontend && sam build)
# `sam deploy` exits non-zero when there's nothing new to deploy - a normal,
# frequent case here (the HTML upload below still needs to happen even when
# the infra itself hasn't changed), not a real failure. Only abort on an
# actual error.
DEPLOY_OUTPUT="$(cd frontend && sam deploy 2>&1)" || {
  if ! grep -q "No changes to deploy" <<< "$DEPLOY_OUTPUT"; then
    echo "$DEPLOY_OUTPUT"
    exit 1
  fi
  echo "No infrastructure changes - continuing to re-upload the dashboard."
}
echo "$DEPLOY_OUTPUT"

echo
echo "== Reading stack outputs =="
API_URL=$(aws cloudformation describe-stacks --stack-name "${BACKEND_STACK}" --region "${REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text)
BUCKET=$(aws cloudformation describe-stacks --stack-name "${FRONTEND_STACK}" --region "${REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='FrontendBucketName'].OutputValue" --output text)
DISTRIBUTION_ID=$(aws cloudformation describe-stacks --stack-name "${FRONTEND_STACK}" --region "${REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='FrontendDistributionId'].OutputValue" --output text)
FRONTEND_URL=$(aws cloudformation describe-stacks --stack-name "${FRONTEND_STACK}" --region "${REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='FrontendUrl'].OutputValue" --output text)

if [ -z "$API_URL" ]; then
  echo "Could not read the backend stack's ApiUrl output - is '${BACKEND_STACK}' deployed?"
  exit 1
fi
echo "API URL:      ${API_URL}"
echo "S3 bucket:    ${BUCKET}"
echo "Distribution: ${DISTRIBUTION_ID}"

echo
echo "== Injecting the API URL into index.html and uploading =="
mkdir -p /tmp/frontend-build
sed "s#REPLACE_WITH_API_URL#${API_URL}#" frontend/index.html > /tmp/frontend-build/index.html
aws s3 cp /tmp/frontend-build/index.html "s3://${BUCKET}/index.html" --region "${REGION}"

echo
echo "== Invalidating the CloudFront cache =="
aws cloudfront create-invalidation --distribution-id "${DISTRIBUTION_ID}" --paths "/*" > /dev/null

echo
echo "== Done =="
echo "Dashboard: ${FRONTEND_URL}"
echo "(CloudFront can take a few minutes to fully propagate on a first deploy)"
