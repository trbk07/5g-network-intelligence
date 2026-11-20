"""Capture actual running UI, API docs and a 100-second captioned replay walkthrough."""

import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    target = Path("docs/screenshots")
    target.mkdir(parents=True, exist_ok=True)
    videos = Path(".cache/demo-video")
    videos.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 1200},
            record_video_dir=str(videos),
            record_video_size={"width": 1440, "height": 1200},
        )
        page = context.new_page()
        page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded")
        page.get_by_text("Risk detail", exact=True).wait_for(timeout=45000)
        page.locator(".js-plotly-plot").wait_for(timeout=15000)
        page.screenshot(path=str(target / "dashboard.png"), full_page=True)

        def caption(text):
            page.evaluate(
                """text => {
                let banner=document.getElementById('walkthrough-caption');
                if(!banner){banner=document.createElement('div');banner.id='walkthrough-caption';document.body.appendChild(banner);}
                banner.style.cssText='position:fixed;bottom:18px;left:335px;right:28px;background:#142632;color:white;padding:16px 22px;border-radius:8px;z-index:999999;font:18px Arial;box-shadow:0 3px 16px #0003';
                banner.textContent=text;
            }""",
                text,
            )

        caption(
            "5G Network Intelligence: forecast beam usage at +1, +6 and +24 hours. Historical offline replay, not live telemetry."
        )
        page.wait_for_timeout(16000)
        caption(
            "Explore 24 real beams from NETOP GOTS-F. The chart contains 168 observed hours and three point forecasts."
        )
        page.get_by_test_id("stSelectbox").nth(2).click()
        page.get_by_role("option", name="0_0_8", exact=True).click()
        page.wait_for_timeout(16000)
        caption(
            "Change the forecast origin. The fixed model uses observations through that origin; known targets are displayed only after prediction."
        )
        slider = page.get_by_role("slider")
        slider.focus()
        for _ in range(24):
            slider.press("ArrowRight")
        page.wait_for_timeout(18000)
        caption(
            "Historical P90/P95 are beam thresholds, not forecast confidence bands or congestion probabilities."
        )
        page.get_by_text("Risk detail", exact=True).scroll_into_view_if_needed()
        page.wait_for_timeout(16000)
        caption(
            "The comparison panel shows the frozen full-network holdout: about 23.5% lower MAE than validation-selected baselines."
        )
        page.wait_for_timeout(16000)
        caption(
            "Limitations remain visible: P90 recall is only about 10-16%. Many exceedances are missed. No production or incident-detection claim."
        )
        page.wait_for_timeout(18000)
        video = page.video
        page.close()
        context.close()
        shutil.copyfile(video.path(), "docs/demo_walkthrough.webm")
        docs = browser.new_page(viewport={"width": 1440, "height": 1100})
        docs.goto("http://127.0.0.1:8000/docs", wait_until="networkidle")
        docs.get_by_text("5G Network Intelligence", exact=False).first.wait_for()
        docs.screenshot(path=str(target / "api_docs.png"), full_page=True)
        browser.close()
    print("Actual dashboard/API screenshots and captioned 100-second browser video saved")


if __name__ == "__main__":
    main()
