import base64
from multiprocessing.sharedctypes import Value

import cv2

import pysoarlib as psl
import Python_sml_ClientInterface as sml


class AtariConnector(psl.AgentConnector):
    def __init__(self, agent):
        super().__init__(agent)
        self.agent = agent
        self.gui = None
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

    def on_output_event(self, command_name, root_id):
        if command_name == "take-action":
            action_id = int(root_id.GetParameterValue("action-id"))
        if self.gui is not None:
            self.gui.step_env(action_id)
            root_id.AddStatusComplete()
        return super().on_output_event(command_name, root_id)


class StateViewerConnector(psl.AgentConnector):
    def __init__(self, agent):
        super().__init__(agent)
        self.agent = agent
        self.gui = None
    
    def on_input_phase(self, input_link):
        state_text = self.agent.execute_command("p S1 -d 7", True)
        self.gui.soar_state_viewer_callback(state_text)
        # print(state_text)
