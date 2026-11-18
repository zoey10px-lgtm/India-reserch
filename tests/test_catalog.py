import json
import tempfile
import unittest
from pathlib import Path
from research.catalog import resolve_catalog_model

class CatalogTests(unittest.TestCase):
    def test_declared_model_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "org" / "encoder" / "rev"
            model.mkdir(parents=True)
            catalog = root / "model-catalog.json"
            catalog.write_text(json.dumps({"models": [
                {"catalog_id": "enc", "org": "org", "name": "encoder", "revision": "rev"}
            ]}), encoding="utf-8")
            self.assertEqual(resolve_catalog_model("enc", catalog, root), model)
            with self.assertRaises(ValueError):
                resolve_catalog_model("unapproved", catalog, root)

    def test_reject_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "model-catalog.json"
            catalog.write_text(json.dumps({"models": [
                {"catalog_id": "unsafe", "store_path": "../outside"}
            ]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_catalog_model("unsafe", catalog, root)

if __name__ == "__main__":
    unittest.main()
