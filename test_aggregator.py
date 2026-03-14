import json
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from Dashboard.geospatial_aggregator import GeoSpatialAggregator, LocationResolver

with open("mock_geo_tweets.json", "r", encoding="utf-8") as f:
    tweets = json.load(f)

print(f"Loaded {len(tweets)} tweets")

aggregator = GeoSpatialAggregator(min_cluster_size=1)
reports = aggregator.analyze_all_clusters(tweets)

print(f"Generated {len(reports)} cluster reports\n")

for r in reports:
    print(f"  {r['location']:30s} {r['total_tweets']:>4} tweets  "
          f"{r['unique_authors']:>3} authors  {r['status']:30s} {r['severity']}")

# Show one detailed report
for r in reports:
    if r["status"] == "confirmed_event":
        print(f"\nDetailed: {r['location']}")
        print(f"  Primary: {r['primary_label']} ({r['agreement_score']:.0%} agreement)")
        for lid, info in r["label_distribution"].items():
            print(f"    {info['label_name']:25s} {info['count']:>3} ({info['percentage']:>5.1f}%)")
        print(f"  Actions:")
        for a in r["recommended_actions"]:
            print(f"    {a}")
        break
