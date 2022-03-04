import base64

import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

import gym


ALL_GYM_ENVS = list(k for k in gym.envs.registry.env_specs.keys())
SOAR_ALE_ENVS = ("Breakout-v0", "Breakout-v4")


class SoarAleGui(tk.Tk):
    def __init__(self, atari_agent, atari_connector, state_viewer_connector):
        super().__init__()
        
        self.title = "Soar-ALE Experiment Platform"
        self.frame_time_ms = 16

        self.agent = atari_agent
        self.connector = atari_connector
        self.state_viewer_connector = state_viewer_connector

        self.ale_game_selection = tk.StringVar()
        self.ale_action_str = tk.StringVar(value="Action Taken: ")
        self.ale_last_reward_str = tk.StringVar(value="Last Reward: ")
        self.ale_total_reward_str = tk.StringVar(value="Total Reward: ")
        self.ale_playing_str = tk.StringVar(value="Playing: ")

        self.current_environment = None
        self.current_observation = None
        self.current_reward = 0
        self.cumulative_reward = 0
        self.playing = False

        self.ale_photoimage = None
        self.ale_canvas_image = None
        self.next_action = None

        self.soar_user_input_str = tk.StringVar()

        self.make_ale_frame()
        self.make_soar_frame()

###############################
# ARCADE LEARNING ENVIRONMENT #
###############################
# GUI STUFF
###########

    def make_ale_frame(self):
        # Create ALE frame and associated widgets
        self.ale_frame = ttk.Frame(self)
        self.ale_action_label = ttk.Label(self.ale_frame, textvariable=self.ale_action_str)
        self.ale_last_reward_label = ttk.Label(self.ale_frame, textvariable=self.ale_last_reward_str)
        self.ale_total_reward_label = ttk.Label(self.ale_frame, textvariable=self.ale_total_reward_str)
        self.ale_playing_label = ttk.Label(self.ale_frame, textvariable=self.ale_playing_str)
        self.ale_view_canvas = tk.Canvas(self.ale_frame)
        self.ale_game_select_dropdown = ttk.Combobox(self.ale_frame, textvariable=self.ale_game_selection)
        self.ale_reset_button = ttk.Button(self.ale_frame, text="Reset", command=self.ale_game_reset_callback)
        self.ale_play_button = ttk.Button(self.ale_frame, text="Play", command=self.ale_game_play_callback)
        self.ale_pause_button = ttk.Button(self.ale_frame, text="Pause", command=self.ale_game_pause_callback)
        self.ale_game_select_dropdown['values'] = SOAR_ALE_ENVS
        self.ale_game_select_dropdown.state(["readonly"])
        self.ale_game_select_dropdown.set(SOAR_ALE_ENVS[0])

        # Place ALE widgets in grid
        self.ale_frame.grid(column=0, row=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.ale_action_label.grid(column=0, row=0)
        self.ale_last_reward_label.grid(column=0, row=1)
        self.ale_total_reward_label.grid(column=0, row=2)
        self.ale_playing_label.grid(column=0, row=3)
        self.ale_view_canvas.grid(column=1, row=0, rowspan=4, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.ale_game_select_dropdown.grid(column=2, row=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.ale_reset_button.grid(column=2, row=1, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.ale_play_button.grid(column=2, row=2, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.ale_pause_button.grid(column=2, row=3, sticky=(tk.N, tk.S, tk.W, tk.E))

    def ale_game_reset_callback(self):
        self.current_environment = gym.make(self.ale_game_selection.get())
        self.update_observation(self.current_environment.reset())

    def ale_game_play_callback(self):
        if self.playing:
            return
        self.playing = True
        self.after(self.frame_time_ms, self.run_env_randomly)

    def ale_game_pause_callback(self):
        self.playing = False

# GYM RUNNING STUFF
###################
    def step_env(self, action):
        obs, rwrd, done, info = self.current_environment.step(action)
        self.update_observation(obs)
        self.current_reward = rwrd
        self.cumulative_reward += rwrd
        if done:
            self.playing = False

        self.ale_action_str.set(f"Action: {action}")
        self.ale_last_reward_str.set(f"Last reward: {rwrd}")
        self.ale_total_reward_str.set(f"Total reward: {self.cumulative_reward}")
        self.ale_playing_str.set(f"Playing: {self.playing}")

    def run_env_randomly(self):
        if not self.playing:
            return
        if self.next_action is None:
            action = self.current_environment.action_space.sample()
        else:
            action = self.next_action
        self.step_env(action)
        self.after(self.frame_time_ms, self.run_env_randomly)

    def update_observation(self, observation):
        self.current_observation = observation
        if self.ale_canvas_image is not None:
            self.ale_view_canvas.delete(self.ale_canvas_image)
        self.ale_tk_image = ImageTk.PhotoImage(Image.fromarray(observation))
        self.ale_canvas_image = self.ale_view_canvas.create_image(0, 0, image=self.ale_tk_image, anchor='nw')
        
        observation_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        cv2.imwrite("../observation_raw.png", observation_bgr)
        self.connector.send_vision(observation_bgr)

##############
# SOAR AGENT #
##############
# GUI STUFF
##############

    def make_soar_frame(self):
        # Create soar frame and widgets
        self.soar_frame = ttk.Frame(self)
        self.soar_output_frame = ttk.Frame(self.soar_frame)
        self.soar_output_text = tk.Text(self.soar_output_frame, width=100, height=100)
        self.soar_output_scrollbar = ttk.Scrollbar(self.soar_output_frame, 
                                                   orient=tk.VERTICAL,
                                                   command=self.soar_output_text.yview)
        self.soar_output_text.configure(yscrollcommand=self.soar_output_scrollbar.set)
        
        self.soar_input_frame = ttk.Frame(self.soar_frame)
        self.soar_input_entry = ttk.Entry(self.soar_input_frame, textvariable=self.soar_user_input_str)
        self.soar_input_send_button = ttk.Button(self.soar_input_frame, text="Send", command=self.soar_input_send_callback)
        self.soar_input_step_button = ttk.Button(self.soar_input_frame, text="Step", command=self.soar_input_step_callback)
        self.soar_input_print_state_button = ttk.Button(self.soar_input_frame, text="State", command=self.soar_input_print_state_callback)

        self.soar_state_viewer_frame = ttk.Frame(self.soar_frame)
        self.soar_state_viewer_text = tk.Text(self.soar_state_viewer_frame, width=80, height=100)

        # Place frame and widgets in grid
        self.soar_frame.grid(column=1, row=0, sticky=tk.NSEW)
        self.soar_output_frame.grid(column=0, row=0, sticky=tk.NSEW)
        self.soar_output_text.grid(column=0, row=0, stick=tk.NSEW)
        self.soar_output_scrollbar.grid(column=1, row=0, sticky=tk.NSEW)

        self.soar_input_frame.grid(column=0, row=1, sticky=tk.NSEW)
        self.soar_input_entry.grid(column=0, row=0, sticky=tk.NSEW)
        self.soar_input_send_button.grid(column=1, row=0, sticky=tk.NSEW)
        self.soar_input_step_button.grid(column=2, row=0, sticky=tk.NSEW)
        self.soar_input_print_state_button.grid(column=3, row=0, sticky=tk.NSEW)
        
        self.soar_state_viewer_frame.grid(column=1, row=0, sticky=tk.NSEW)
        self.soar_state_viewer_text.grid(column=0, row=0, stick=tk.NSEW)

    def soar_output_callback(self, text):
        self.soar_output_text.insert(tk.END, text)
        self.soar_output_text.insert(tk.END, "\n")
        print(text)

    def soar_input_send_callback(self):
        self.agent.execute_command(self.soar_user_input_str.get(), True)

    def soar_input_step_callback(self):
        self.agent.execute_command("step", True)

    def soar_input_print_state_callback(self):
        self.agent.execute_command("p S1 -d 7", True)

    def soar_state_viewer_callback(self, state_text):
        self.soar_state_viewer_text.delete(1.0, tk.END)
        self.soar_state_viewer_text.insert(tk.END, state_text)
        self.soar_state_viewer_text.insert(tk.END, "\n")
