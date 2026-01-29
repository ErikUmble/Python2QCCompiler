import datetime
from benchmarklib.core.database import BackendPropertyManager, logger as db_logger
import logging
from qiskit_ibm_runtime import QiskitRuntimeService

db_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
db_logger.addHandler(handler)
db_logger.propagate = False


service = QiskitRuntimeService()
backend = service.backend("ibm_rensselaer")

db = BackendPropertyManager("rbf.db")

db.load_missing_dates(backend, start_date=datetime.date(2025, 1, 1))