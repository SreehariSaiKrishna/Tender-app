#!/usr/bin/env bash
# Add, remove and fix up dashboard logins (the backend stack's Cognito user
# pool). There's no public sign-up - this is the only way in.
#
#   ./scripts/manage-users.sh add       person@company.com
#   ./scripts/manage-users.sh disable   person@company.com   # offboard: blocks login + signs out everywhere
#   ./scripts/manage-users.sh enable    person@company.com
#   ./scripts/manage-users.sh reset     person@company.com   # forgot password
#   ./scripts/manage-users.sh new-mfa   person@company.com   # lost/new phone: re-create the login
#   ./scripts/manage-users.sh delete    person@company.com
#   ./scripts/manage-users.sh list
#
# `add` makes Cognito email the person a temporary password (valid 3 days).
# On first login they choose their own password and must set up an
# authenticator app (Google/Microsoft Authenticator etc) - MFA is mandatory.
set -euo pipefail

export AWS_PROFILE="${AWS_PROFILE:-tender-agent}"
REGION="ap-south-1"
BACKEND_STACK="tender-agent"

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

ACTION="${1:-}"
EMAIL="${2:-}"
[ -n "$ACTION" ] || usage
if [ "$ACTION" != "list" ] && [ -z "$EMAIL" ]; then usage; fi

POOL_ID=$(aws cloudformation describe-stacks --stack-name "${BACKEND_STACK}" --region "${REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='UserPoolId'].OutputValue" --output text)
if [ -z "$POOL_ID" ] || [ "$POOL_ID" = "None" ]; then
  echo "Could not read UserPoolId from the '${BACKEND_STACK}' stack - deploy the backend first (./scripts/deploy.sh)."
  exit 1
fi

cognito() { aws cognito-idp "$@" --user-pool-id "${POOL_ID}" --region "${REGION}"; }

case "$ACTION" in
  add)
    cognito admin-create-user --username "${EMAIL}" \
      --user-attributes "Name=email,Value=${EMAIL}" "Name=email_verified,Value=true" \
      --desired-delivery-mediums EMAIL > /dev/null
    echo "Created ${EMAIL} - a temporary password has been emailed to them."
    ;;
  disable)
    cognito admin-disable-user --username "${EMAIL}"
    # Revokes their refresh tokens so no open tab can renew; an access token
    # already issued stays valid until it expires (at most 1 hour).
    cognito admin-user-global-sign-out --username "${EMAIL}"
    echo "Disabled ${EMAIL} and signed them out everywhere."
    ;;
  enable)
    cognito admin-enable-user --username "${EMAIL}"
    echo "Re-enabled ${EMAIL}."
    ;;
  reset)
    cognito admin-reset-user-password --username "${EMAIL}"
    cognito admin-user-global-sign-out --username "${EMAIL}"
    echo "Reset ${EMAIL}: they'll get a code by email to set a new password at next login."
    ;;
  new-mfa)
    # With MFA mandatory, Cognito keeps challenging for the old authenticator
    # even if it's switched off per user - re-creating the login is the
    # reliable way to let them enrol a new one.
    cognito admin-delete-user --username "${EMAIL}"
    cognito admin-create-user --username "${EMAIL}" \
      --user-attributes "Name=email,Value=${EMAIL}" "Name=email_verified,Value=true" \
      --desired-delivery-mediums EMAIL > /dev/null
    echo "Re-created ${EMAIL} - a new temporary password has been emailed; they'll set up their authenticator again."
    ;;
  delete)
    cognito admin-delete-user --username "${EMAIL}"
    echo "Deleted ${EMAIL}."
    ;;
  list)
    cognito list-users \
      --query "Users[].[Attributes[?Name=='email']|[0].Value, UserStatus, Enabled]" --output table
    ;;
  *)
    usage
    ;;
esac
