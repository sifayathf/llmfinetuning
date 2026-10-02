"""Create a small, fictional call-center troubleshooting PDF for testing the pipeline.

    python scripts/make_sample_pdf.py   ->  data/sample_troubleshooting_guide.pdf

Replace it with your real company PDF when you're ready.
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_troubleshooting_guide.pdf"

INTRO = [
    ("1. About this guide",
     "This guide is used by NimbusNet Broadband call-center agents (Tier 1 support) to troubleshoot "
     "home internet issues on the NimbusNet HX-200 router. Support hours are 7 AM to 11 PM, seven days "
     "a week. The customer support phone number is 1-800-555-0199."),
    ("2. Call handling policy",
     "Agents must verify the customer's identity with the account number and the last four digits of "
     "the registered phone number before making any change. The target average handle time is 6 minutes. "
     "Every call must be logged in the CRM with a disposition code. If an issue is not resolved within "
     "20 minutes, the agent must escalate the ticket to Tier 2 support."),
    ("3. Refund and credit policy",
     "Customers are eligible for a service credit when an outage lasts more than 24 hours. The credit is "
     "one day of service for every 24 hours of outage. Tier 1 agents can approve credits up to 25 dollars; "
     "anything above 25 dollars requires supervisor approval. Refunds are processed within 7 to 10 business days."),
]

ISSUES = [
    ("Issue 1: No internet connection (red LED)",
     "The router's Internet LED is solid red and no devices can browse.",
     "The router cannot reach the NimbusNet network, usually because of an area outage or a loose WAN cable.",
     ["Check the outage map in the CRM for the customer's postcode.",
      "Ask the customer to confirm the grey WAN cable is firmly connected to the port labelled WAN.",
      "Power-cycle the router: unplug it for 30 seconds, then plug it back in and wait 3 minutes.",
      "If the LED is still red, run a remote line test from the CRM. If the line test fails, book a technician visit."]),
    ("Issue 2: Slow internet speed",
     "Speed tests show less than 50 percent of the subscribed plan speed.",
     "Wi-Fi interference, too many connected devices, or the customer testing over Wi-Fi far from the router.",
     ["Ask the customer to run a speed test at speedtest.nimbusnet.example with a device connected by Ethernet cable.",
      "If the wired speed is normal, the issue is Wi-Fi: move the router to a central, elevated location.",
      "Change the Wi-Fi channel from the router admin page at 192.168.20.1 under Wireless > Channel.",
      "If the wired speed is also slow, run a remote line test and escalate to Tier 2 if it shows errors."]),
    ("Issue 3: Wi-Fi keeps disconnecting",
     "Devices drop the Wi-Fi connection every few minutes.",
     "Outdated router firmware or band steering problems with older devices.",
     ["Check the firmware version in the CRM. The current firmware version is 4.2.7.",
      "If the firmware is older than 4.2.7, push a firmware update remotely from the CRM.",
      "Ask the customer to disable Smart Connect (band steering) under Wireless > Advanced.",
      "Advise connecting older devices to the 2.4 GHz network named with the suffix -2G."]),
    ("Issue 4: Forgot Wi-Fi password",
     "The customer cannot connect new devices because they do not know the Wi-Fi password.",
     "The default password was changed and forgotten.",
     ["The default Wi-Fi password is printed on the sticker on the bottom of the router.",
      "If the customer changed it, they can view or reset it in the NimbusNet app under My Network > Wi-Fi Settings.",
      "As a last resort, a factory reset restores the default password: hold the reset button for 15 seconds."]),
    ("Issue 5: Error code E-102",
     "The router display or app shows error code E-102.",
     "E-102 means PPPoE authentication failed, usually because of an incorrect username or a suspended account.",
     ["Check that the account is active and not suspended for non-payment in the billing system.",
      "Re-provision the PPPoE credentials remotely from the CRM using the Re-sync Credentials button.",
      "Ask the customer to restart the router. If E-102 persists, escalate to Tier 2 with the error code."]),
    ("Issue 6: Error code E-307",
     "The router shows error code E-307 and the Power LED blinks amber.",
     "E-307 indicates the router is overheating.",
     ["Ask the customer to move the router to a well-ventilated area away from other electronics.",
      "Make sure the router stands vertically on its stand and the vents are not blocked.",
      "If E-307 appears again within 48 hours, arrange a free hardware replacement."]),
    ("Issue 7: Router keeps rebooting",
     "The router restarts by itself several times a day.",
     "A faulty power adapter or a corrupted firmware installation.",
     ["Confirm the customer is using the original 12V 2A NimbusNet power adapter.",
      "Push a firmware re-install from the CRM.",
      "If reboots continue after the re-install, the router is faulty: send a replacement HX-200 by courier within 2 business days."]),
    ("Issue 8: Cannot access router admin page",
     "The customer cannot open the router configuration page in the browser.",
     "Wrong address, or the device is not connected to the router's network.",
     ["The admin page address is 192.168.20.1 and the default username is admin.",
      "The default admin password is printed on the router sticker, labelled Admin Key.",
      "Make sure the device is connected to the NimbusNet Wi-Fi or by Ethernet, and not on mobile data."]),
    ("Issue 9: Port forwarding for gaming",
     "Online games report a strict NAT type.",
     "Required ports are not forwarded, or UPnP is disabled.",
     ["Enable UPnP on the admin page under Advanced > NAT.",
      "If needed, add port forwarding rules under Advanced > Port Forwarding.",
      "Customers with CGNAT need a static public IP add-on, which costs 5 dollars per month."]),
    ("Issue 10: Intermittent dropouts every evening",
     "The connection drops for a few minutes, mostly between 7 PM and 11 PM.",
     "Network congestion in the area or a degraded line.",
     ["Check the CRM line history graph for SNR drops in the evening.",
      "If the SNR margin is below 6 dB, book a technician to inspect the line.",
      "If the area is flagged as congested, inform the customer that a capacity upgrade is planned and log the ticket with code CONG-01."]),
]

ESCALATION = [
    ("5. Escalation matrix",
     "Tier 1 handles basic troubleshooting. Tier 2 handles line faults, firmware failures and repeated "
     "error codes. Network Operations Center (NOC) handles area outages affecting more than 50 customers. "
     "Escalations to Tier 2 must include the account number, error codes, and the steps already performed. "
     "Priority P1 tickets (business customers with a full outage) must be answered within 1 hour."),
    ("6. Technician visits",
     "Technician visits are free when the fault is on the NimbusNet network. If the fault is caused by the "
     "customer's internal wiring, a call-out fee of 60 dollars applies. Visits can be booked from 8 AM to 6 PM "
     "Monday to Saturday."),
]


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ss = getSampleStyleSheet()
    story = [Paragraph("NimbusNet Broadband - Tier 1 Troubleshooting Guide (SAMPLE)", ss["Title"]), Spacer(1, 12)]
    for title, body in INTRO:
        story += [Paragraph(title, ss["Heading2"]), Paragraph(body, ss["BodyText"]), Spacer(1, 8)]
    story += [PageBreak(), Paragraph("4. Troubleshooting procedures", ss["Heading1"])]
    for title, symptoms, cause, steps in ISSUES:
        story += [
            Paragraph(title, ss["Heading2"]),
            Paragraph(f"<b>Symptoms:</b> {symptoms}", ss["BodyText"]),
            Paragraph(f"<b>Cause:</b> {cause}", ss["BodyText"]),
            Paragraph("<b>Resolution:</b>", ss["BodyText"]),
        ]
        story += [Paragraph(f"{i}. {s}", ss["BodyText"]) for i, s in enumerate(steps, 1)]
        story.append(Spacer(1, 10))
    story.append(PageBreak())
    for title, body in ESCALATION:
        story += [Paragraph(title, ss["Heading2"]), Paragraph(body, ss["BodyText"]), Spacer(1, 8)]
    SimpleDocTemplate(str(OUT), pagesize=A4, title="NimbusNet Troubleshooting Guide").build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
