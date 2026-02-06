import os

from tminterface2 import TMInterface, MessageType
import struct

import time

PORT = 8477  # must match `custom_port`


import subprocess
import threading
from pathlib import Path

TM_EXE = Path(os.environ["TMNF_DIR"]) / "TmForever.exe"

#subprocess.Popen(
#    ["wine", str(TM_EXE)]
#)
def popen_and_call(on_exit, *popen_args):
    """
    Runs the given args in a subprocess.Popen, and then calls the function
    on_exit when the subprocess completes.
    on_exit is a callable object, and popen_args is a list/tuple of args that
    would give to subprocess.Popen.
    """
    def run_in_thread(on_exit, *popen_args):
        proc = subprocess.Popen(*popen_args)
        proc.wait()
        on_exit()
        return
    thread = threading.Thread(target=run_in_thread, args=(on_exit, popen_args))
    thread.start()
    # returns immediately after the thread starts
    return thread

def exit_handler():
    print('TMNF Exited')

popen_and_call(exit_handler, "strace", "wine", TM_EXE)

time.sleep(15)
print(subprocess.check_output(["netstat", "-apn"]))

def main():
    tm = TMInterface(PORT)
    tm.register()

    print("Agent connected, waiting for simulation steps")

    sock = tm.sock  # direct access is intentional here

    tm.execute_command('map r"A01-Race.Challenge.Gbx"')
    while True:
        # --- wait for server message ---
        msg_type = struct.unpack("i", sock.recv(4))[0]

        if msg_type == MessageType.SC_RUN_STEP_SYNC:
            # read race time (int)
            race_time = struct.unpack("i", sock.recv(4))[0]

            # acknowledge step (THIS UNBLOCKS THE SIM)
            sock.sendall(struct.pack("i", MessageType.SC_RUN_STEP_SYNC))

            # always press gas
            tm.set_input_state(
                left=False,
                right=False,
                accelerate=True,
                brake=False
            )

        elif msg_type == MessageType.SC_ON_CONNECT_SYNC:
            sock.sendall(struct.pack("i", MessageType.SC_ON_CONNECT_SYNC))

        else:
            # ignore other sync events safely
            sock.sendall(struct.pack("i", msg_type))


if __name__ == "__main__":
    main()
