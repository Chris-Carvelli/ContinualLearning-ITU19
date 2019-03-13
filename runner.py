from src.ga import GA
from sessions.session import Session, MultiSession

# Example of MultiSession
config_one = 'config_files/config_one'
config_fast = 'config_files/config_faster'
ms = MultiSession([GA(config_fast), GA(config_fast), GA(config_fast)], "TestSessions")
ms.start()