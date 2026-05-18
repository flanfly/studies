from playwright.sync_api import sync_playwright

LOGIN_URL = "https://www.polaritydigital.io/"
API_URL = "api.polaritydigital.io/api/getSubscription"


def launch_browser(p, headless=True):
    return p.chromium.launch_persistent_context(
        user_data_dir="./browser_profile",
        channel="chrome",
        headless=headless,
        args=["--disable-blink-features=AutomationControlled"],
    )


def login():
    with sync_playwright() as p:
        context = launch_browser(p, headless=False)
        page = context.new_page()
        page.goto(LOGIN_URL)

        input("Press Enter here in the terminal AFTER you have fully logged in...")


def fetch_bearer_token():
    bearer_token = None

    def intercept_request(request):
        nonlocal bearer_token

        if API_URL in request.url:
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                bearer_token = auth_header

    with sync_playwright() as p:
        context = launch_browser(p, headless=True)
        page = context.new_page()
        page.on("request", intercept_request)

        page.goto(LOGIN_URL)
        page.locator("div.ant-space-item:has-text('kai.h.michaelis')").first.wait_for()
        page.wait_for_load_state("networkidle")

        context.close()
        return bearer_token


token = fetch_bearer_token()
if token is None:
    print("failed to extract token")
else:
    with open("IDTOKEN", "w") as f:
        f.write(token)
    print("saved token to IDTOKEN")
