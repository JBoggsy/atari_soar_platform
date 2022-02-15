import base64

import cv2

import pysoarlib as psl
import Python_sml_ClientInterface as sml


class AtariConnector(psl.AgentConnector):
    def __init__(self, agent):
        super().__init__(agent)
        self.agent = agent
        self.agent.execute_command("svs --enable")

        self.new_vision_update = False
        self.vision_update_num = 0
        self.new_vision_update_wme = psl.SoarWME("vision-update", self.vision_update_num)

    def send_vision(self, visual):
        success, data = cv2.imencode('.png', visual)
        obs_data_b64 = base64.b64encode(data).decode()
        inject_cmd_str = f"svs vsm.inject {obs_data_b64}"
        self.agent.execute_command(inject_cmd_str)
        self.new_vision_update = True
        self.vision_update_num += 1

    def on_input_phase(self, input_link):
        if self.new_vision_update:
            self.new_vision_update_wme.set_value(self.vision_update_num)
            self.new_vision_update_wme.update_wm(input_link)
            self.new_vision_update = False


class StateViewerConnector(psl.AgentConnector):
    def __init__(self, agent):
        super().__init__(agent)
        self.agent = agent
        self.gui = None

    def add_gui(self, gui):
        self.gui = gui
    
    def on_input_phase(self, input_link):
        state_text = self.agent.execute_command("p S1 -d 7", True)
        self.gui.soar_state_viewer_callback(state_text)
        print(state_text)
