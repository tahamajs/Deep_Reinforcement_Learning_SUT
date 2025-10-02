import sys
import os

sys.path.insert(0, "/Users/tahamajs/Documents/uni/DRL/CAs/Solutions/CA17")

try:
    from models.world_models import RSSMCore

    print("✅ RSSMCore import successful")
except Exception as e:
    print(f"❌ RSSMCore import failed: {e}")

try:
    from models import RSSMCore

    print("✅ models.RSSMCore import successful")
except Exception as e:
    print(f"❌ models.RSSMCore import failed: {e}")