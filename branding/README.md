OAKS branding for Cognito's hosted login page - the page the dashboard's
"Sign in with your OAKS account" button opens. `scripts/deploy.sh` applies
both files after every backend deploy (`aws cognito-idp set-ui-customization`).

- `oaks-wordmark.png` - the logo shown in the page's banner. Cognito limits
  it to PNG/JPG under 100 KB. It's white on transparent, so the banner is navy.
- `cognito-login.css` - colours and sizes. Cognito accepts only its own
  `*-customizable` classes and a limited set of properties (no images,
  gradients or web fonts); anything else makes the deploy step fail.

The dashboard's own sign-in screen (frontend/index.html, `#signinScreen`)
carries the full design; this only keeps the hosted page in the same colours.
