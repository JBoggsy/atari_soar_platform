import tkinter as tk
import gym
import cv2
from gui import SoarAleGui

import pysoarlib as psl
from agent_connector import AtariConnector, StateViewerConnector

if __name__ == "__main__":
    agent = psl.SoarClient(agent_name="atari",
                           agent_source="/home/boggsj/Coding/research/atari_soar/atari_agents/agent_1/agent_1.soar",
                           write_to_stdout=True,
                           watch_level=4)
    
    atari_connector = AtariConnector(agent)
    agent.add_connector("atari", atari_connector)

    state_view_connector = StateViewerConnector(agent)
    agent.add_connector("state_viewer", state_view_connector)

    gui = SoarAleGui(atari_agent=agent, atari_connector=atari_connector, state_viewer_connector=state_view_connector)
    state_view_connector.add_gui(gui)
    
    agent.print_handler=gui.soar_output_callback
    agent.connect()
    gui.mainloop()