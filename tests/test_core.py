import unittest
from backend.pipeline import generate_demo,clean,build_db,train
from fastapi.testclient import TestClient
from backend.main import app
class TestCore(unittest.TestCase):
    def test_synthetic_and_clean(self):
        d=clean(generate_demo(450));self.assertEqual(len(d),450);self.assertTrue(set(d.congestion_level).issubset({'Low','Moderate','High'}))
    def test_endpoints(self):
        with TestClient(app) as c:
            self.assertEqual(c.get('/api/health').status_code,200)
            self.assertGreater(c.get('/api/overview').json()['summary']['observations'],0)
            self.assertEqual(c.get('/api/olap?group_by=road').status_code,200)
            r=c.post('/api/predict',json={'area':'Sion','road':'LBS Marg','hour':18,'vehicle_count':190,'avg_speed_kmh':23});self.assertEqual(r.status_code,200);self.assertIn(r.json()['prediction'],['Low','Moderate','High'])
if __name__=='__main__': unittest.main()
