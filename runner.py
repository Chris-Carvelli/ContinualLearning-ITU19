from src.ga import GA
from sessions.session import Session, MultiSession

# Example of MultiSession
ms = MultiSession([GA('config_files/config_one'), GA('config_files/config_one'), GA('config_files/config_one')], "TestSessions")
ms.start()