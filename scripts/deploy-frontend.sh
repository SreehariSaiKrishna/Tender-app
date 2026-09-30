#!/usr/bin/env bash
# Deploy the Tender Intelligence Agent frontend (S3 + CloudFront).
#
# Deliberately a separate script from scripts/deploy.sh (the backend) - the
# two stacks are unrelated infrastructure. The only place they connect is
# here: this script reads the *backend* stack's ApiUrl and Cognito outputs
# and bakes them into index.html as plain static strings before uploading
# (and passes the same URLs to the frontend stack for its CSP header) - not
# a CloudFormation cross-stack reference, just a deploy-time convenience.
#
# Needs the backend deployed first (./scripts/deploy.sh), and the backend in
# turn reads this stack's FrontendUrl for CORS and the login redirect - so
# on a brand-new account the frontend stack has to be created once by hand
# (cd frontend && sam deploy) before either script will run.
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
echo "== Reading the backend stack's outputs =="
backend_output() {
  aws cloudformation describe-stacks --stack-name "${BACKEND_STACK}" --region "${REGION}"     --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
API_URL=$(backend_output ApiUrl)
COGNITO_DOMAIN=$(backend_output CognitoDomain)
COGNITO_CLIENT_ID=$(backend_output UserPoolClientId)
for v in API_URL COGNITO_DOMAIN COGNITO_CLIENT_ID; do
  if [ -z "${!v}" ] || [ "${!v}" = "None" ]; then
    echo "Could not read ${v} from the backend stack - is '${BACKEND_STACK}' deployed with the Cognito login (./scripts/deploy.sh)?"
    exit 1
  fi
done
echo "API URL:      ${API_URL}"
echo "Login page:   ${COGNITO_DOMAIN}"

echo
echo "== Deploying frontend infrastructure (S3 + CloudFront) =="
(cd frontend && sam build)
# `sam deploy` exits non-zero when there's nothing new to deploy - a normal,
# frequent case here (the HTML upload below still needs to happen even when
# the infra itself hasn't changed), not a real failure. Only abort on an
# actual error.
# SAM's output is only shown when something actually happened: its "Error:
# No changes to deploy" line reads like a failure when it isn't one.
if DEPLOY_OUTPUT="$(cd frontend && sam deploy --parameter-overrides "ApiUrl=${API_URL}" "CognitoDomain=${COGNITO_DOMAIN}" 2>&1)"; then
  echo "$DEPLOY_OUTPUT"
elif grep -q "No changes to deploy" <<< "$DEPLOY_OUTPUT"; then
  echo "No infrastructure changes (S3/CloudFront already up to date) - continuing to re-upload the dashboard."
else
  echo "$DEPLOY_OUTPUT"
  exit 1
fi

echo
echo "== Reading the frontend stack's outputs =="
frontend_output() {
  aws cloudformation describe-stacks --stack-name "${FRONTEND_STACK}" --region "${REGION}"     --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
BUCKET=$(frontend_output FrontendBucketName)
DISTRIBUTION_ID=$(frontend_output FrontendDistributionId)
FRONTEND_URL=$(frontend_output FrontendUrl)
echo "S3 bucket:    ${BUCKET}"
echo "Distribution: ${DISTRIBUTION_ID}"

echo
echo "== Injecting the API URL and login settings into index.html and uploading =="
mkdir -p /tmp/frontend-build
sed -e "s#REPLACE_WITH_API_URL#${API_URL}#"     -e "s#REPLACE_WITH_COGNITO_DOMAIN#${COGNITO_DOMAIN}#"     -e "s#REPLACE_WITH_COGNITO_CLIENT_ID#${COGNITO_CLIENT_ID}#"     frontend/index.html > /tmp/frontend-build/index.html
if grep -q "REPLACE_WITH_" /tmp/frontend-build/index.html; then
  echo "index.html still has an unfilled REPLACE_WITH_ placeholder - not uploading."
  exit 1
fi
aws s3 cp /tmp/frontend-build/index.html "s3://${BUCKET}/index.html" --region "${REGION}"

echo
echo "== Invalidating the CloudFront cache =="
aws cloudfront create-invalidation --distribution-id "${DISTRIBUTION_ID}" --paths "/*" > /dev/null

echo
echo "== Done =="
echo "Dashboard: ${FRONTEND_URL}"
echo "(CloudFront can take a few minutes to fully propagate on a first deploy)"
