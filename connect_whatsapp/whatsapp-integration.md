# WhatsApp integration

Connects the existing chatbot ([`generate_prompt/chatbot.py`](../generate_prompt/chatbot.py))
to WhatsApp via Meta's official **WhatsApp Cloud API** (the WhatsApp
Business Platform), so a message sent to your business's WhatsApp number
gets a reply from the bot instead of a person.

```text
Customer's WhatsApp  --->  Meta's servers  --->  POST /webhook  --->  this server
                                                                          |
                                                                    chatbot.py's
                                                                 reply() (same
                                                              logic as the terminal
                                                                    chatbot)
                                                                          |
Customer's WhatsApp  <---  Meta's servers  <---  Graph API "send message" call
```

No reply logic lives in the WhatsApp code -- [`connect_whatsapp/whatsapp_server.py`](whatsapp_server.py)
only does the WhatsApp-specific plumbing (receiving Meta's webhook calls,
sending replies back through Meta's Graph API) and calls straight into the
same `build_engine()`/`reply()` core the terminal chatbot uses. Every prompt
fix from live-testing the terminal bot applies here unchanged.

## Why the Cloud API, not your personal WhatsApp app

This is Meta's own, sanctioned way for a business to connect a bot to
WhatsApp -- no risk of your number being flagged for automation, unlike
unofficial "scan a QR code" libraries that puppet the regular consumer app.
The trade-off is more setup: a Meta Business/Developer account, a phone
number registered specifically for WhatsApp Business (see below), and a
server with a public HTTPS address that's reachable any time a customer
might message.

## 1. Meta setup

1. Create a Meta Developer account at [developers.facebook.com](https://developers.facebook.com/)
   if you don't have one, and a Business account at
   [business.facebook.com](https://business.facebook.com/) if you don't
   have one of those either.
2. At [developers.facebook.com/apps](https://developers.facebook.com/apps),
   create a new app, type **Business**.
3. Add the **WhatsApp** product to the app.
4. Meta gives you a **test phone number** for free during development --
   good enough to build and test everything below. It can only message
   numbers you've added as testers in the app dashboard. When you're ready
   for real customers, add your shop's own number as the WhatsApp Business
   number instead (Meta walks you through verifying it) -- **it can't be a
   number currently active in the regular WhatsApp consumer app**; it has
   to be dedicated to the Business Platform.
5. From the app's WhatsApp > API Setup page, note down:
   - **Phone number ID** -- `WHATSAPP_PHONE_NUMBER_ID`
   - **Temporary access token** shown there is only valid ~24 hours; for
     anything beyond quick testing, generate a **permanent** token instead
     (System Users, under Business Settings > Users > System Users -- create
     one, assign it the WhatsApp app with `whatsapp_business_messaging`
     permission, generate its token) -- `WHATSAPP_ACCESS_TOKEN`
6. From the app's Settings > Basic page, note the **App Secret** (click
   "Show") -- `WHATSAPP_APP_SECRET`. This is used to verify that webhook
   calls actually came from Meta (see `verify_signature()` in
   `whatsapp_server.py`) rather than trust any POST to the URL.
7. Make up your own random string for `WHATSAPP_VERIFY_TOKEN` -- it's not
   from Meta, you choose it and enter the same value in two places (your
   `.env` and the webhook config below) so Meta's one-time setup handshake
   can confirm it's really your server on the other end.

Add all four to your `.env` (never commit or print these):

```
WHATSAPP_ACCESS_TOKEN=...
WHATSAPP_PHONE_NUMBER_ID=...
WHATSAPP_VERIFY_TOKEN=...
WHATSAPP_APP_SECRET=...
```

## 2. Install and run the server

```powershell
.\.venv\Scripts\python.exe -m pip install -r connect_whatsapp\requirements-whatsapp.txt
.\connect_whatsapp\run_whatsapp_server.ps1
```

This starts a Flask server on `0.0.0.0:8000` (override with `-Port`) and
**keeps running** -- it's a server, not a one-shot script. It won't do
anything useful until Meta's webhook is pointed at it (next step), since
nothing calls it locally.

## 3. Get a public HTTPS URL

Meta needs a **public HTTPS URL** for your server's `/webhook` route --
`http://localhost:8000` is not reachable from Meta's side. Pick one path:

**Local testing** -- expose your machine with a tunnel, e.g.
[ngrok](https://ngrok.com/) (`ngrok http 8000`) or
[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
(`cloudflared tunnel --url http://localhost:8000`, no account needed). Either
prints a public `https://...` URL that forwards to your local
`run_whatsapp_server.ps1`. It changes every time you restart the tunnel on
the free tier, so you'll re-enter it in Meta's dashboard each session.

**Render** (a real, always-on deployment) -- this app is ready to deploy as-is:

1. Push this repo to GitHub if it isn't already (Render deploys from a
   connected GitHub repo).
2. In the Render dashboard: **New > Web Service**, connect the repo, pick
   the branch to deploy.
3. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r connect_whatsapp/requirements-whatsapp.txt`
   - **Start Command**: `gunicorn --factory connect_whatsapp.whatsapp_server:make_app --bind 0.0.0.0:$PORT`
     (Render sets `$PORT` itself; `make_app()` in `whatsapp_server.py` is a
     zero-argument factory built exactly for this -- it reads all config
     from environment variables instead of CLI flags, since gunicorn
     imports the module rather than running it as a script.)
   - **Instance type**: the free tier is fine to start.
4. **Environment** tab: add each of the six required variables --
   `PINECONE_API_KEY`, `GEMINI_API_KEY`, `WHATSAPP_ACCESS_TOKEN`,
   `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_VERIFY_TOKEN`, `WHATSAPP_APP_SECRET`
   -- same values as your local `.env` (never commit `.env` itself; Render's
   Environment tab is the equivalent for a deployed service).
5. Deploy. Render gives you a permanent URL like
   `https://your-service.onrender.com` -- use
   `https://your-service.onrender.com/webhook` as the Callback URL below.
6. Render's free tier spins the service down after inactivity and takes a
   few seconds to wake back up on the next request -- the first message
   after a quiet period may reply slowly. An always-on (paid) instance
   avoids that; not a problem worth solving before you've confirmed the
   bot works at all.

Then in the Meta app dashboard, under WhatsApp > Configuration:

1. **Callback URL**: `https://<your-url>/webhook`
2. **Verify token**: the same string you put in `WHATSAPP_VERIFY_TOKEN`
3. Click **Verify and save** -- Meta calls your server's `GET /webhook`
   once to confirm it gets the expected challenge back
   (`verify_webhook()` in `whatsapp_server.py` handles this).
4. Under **Webhook fields**, subscribe to `messages`.

## 4. Test it

Message the test number (or your verified business number) from a phone
that's registered as a tester in the Meta app -- "I want to order a
coffee" should get a reply within a few seconds. Watch the server's log
output for each request; `GET /health` on your server reports how many
conversations are currently in memory.

## Still open

- **In-memory conversation state only**: `conversations` (one per phone
  number) lives in a plain dict inside the running process -- lost on
  restart, and won't work correctly if you ever run more than one worker
  process/replica (each would have its own, inconsistent dict). Fine for a
  single-process demo; a real deployment would move this to a shared store
  (e.g. Redis) keyed by phone number.
- **The 24-hour customer service window**: WhatsApp's own rule, not
  something this code controls -- a business can only send free-form
  replies within 24 hours of the customer's last message. Outside that
  window, only pre-approved *message templates* can be sent, which this
  server doesn't implement. In practice: as long as the customer messaged
  recently, replies work as built; a very delayed reply attempt will fail
  at the Graph API call (logged, not silently lost -- see
  `send_whatsapp_message`'s error handling).
- **Text messages only**: images, voice notes, locations, etc. are
  received and silently ignored (`extract_incoming_text_message` returns
  `None` for anything but `type == "text"`), not answered.
- **No message de-duplication**: Meta can occasionally redeliver the same
  webhook event; a redelivered message currently gets a second reply
  generated and sent rather than being recognized as a repeat.
- **No real ordering/checkout flow**, same limitation as the terminal
  chatbot -- see [customer-assistant-notes.md](../docs/customer-assistant-notes.md).
