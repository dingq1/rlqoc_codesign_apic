# main.py
from multiprocessing import freeze_support

def main():
    # Change the method here as needed, "ppo", "end2end", "sadeadam"
    method = "end2end"

    if method == "ppo":
        from config_ppo import run_ppo
        run_ppo()
    elif method == "end2end":
        from config_end2end import run_end2end
        run_end2end()
    elif method == "sadeadam":
        from config_sadeadam import run_sadeadam
        run_sadeadam()
    else:
        raise ValueError("Unknown method.")

if __name__ == '__main__':
    freeze_support()  # This is necessary on Windows and for some spawn/forkserver environments.
    main()
