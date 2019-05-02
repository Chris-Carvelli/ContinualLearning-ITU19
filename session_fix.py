from sessions.session import MultiSession, load_session

if __name__ == '__main__':
    # name = "copy-size_2-ntm-2-layer_010-x5"
    name = "copy-size_2-ntm-1-layer_012-x5"
    session = load_session(f"Experiments/{name}.ses")

    print(session.worker.workers)
    session = MultiSession(session.worker.workers, name, "succesful", parallel_execution=True)
    session.start()
    session.save_data("session", session._session_data())
    # Remember to copy old log, config and script_copy.py files
