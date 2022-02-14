import tkinter as tk
import gym
import cv2
from gui import SoarAleGui

import pysoarlib as psl
from agent_connector import AtariConnector

if __name__ == "__main__":
    agent = psl.SoarClient(agent_name="atari",
                           agent_source="/home/boggsj/Coding/research/atari_soar/agent_1/agent_1.soar",
                           write_to_stdout=True,
                           watch_level=4)
    connector = AtariConnector(agent)
    agent.add_connector(connector, "atari")
    gui = SoarAleGui(atari_agent=agent, atari_connector=connector)
    agent.print_handler=gui.soar_output_callback
    gui.mainloop()