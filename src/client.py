import argparse
import json
import requests
from datetime import datetime, timedelta

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:5001/predict", help="Predict endpoint")
    p.add_argument("--use-occ-dt", action="store_true", help="Send occ_dt (ISO) instead of month/hour/dow")
    p.add_argument("--with-report-delay", type=int, default=None,
                   help="If set (hours), send report_dt = occ_dt + this many hours")
    args = p.parse_args()

    now = datetime.now() - timedelta(hours=2)

    payload = {
        "BIKE_COST": 800,
        "BIKE_TYPE": "City",
        "BIKE_COLOUR": "Black",
        "DIVISION": "D14",
        "LOCATION_TYPE": "Street",
        "PREMISES_TYPE": "Commercial",
        "PRIMARY_OFFENCE": "Theft"
    }

    if args.use_occ_dt:
        occ_dt_str = now.strftime("%Y-%m-%dT%H:%M")
        payload["occ_dt"] = occ_dt_str
        if args.with-report-delay is not None:
            rep_dt = now + timedelta(hours=args.with_report_delay)
            payload["report_dt"] = rep_dt.strftime("%Y-%m-%dT%H:%M")
    else:
        payload["occ_month"] = now.month
        payload["occ_hour"]  = now.hour
        payload["occ_dow"]   = now.weekday()  # Monday=0..Sunday=6
        # If you want to test report delay without occ_dt, omit it (API will use DEFAULT_REPORT_DELAY_H)

    print("POST", args.url)
    print(json.dumps(payload, indent=2))
    r = requests.post(args.url, json=payload, timeout=20)
    print("\nResponse:", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)

if __name__ == "__main__":
    main()
