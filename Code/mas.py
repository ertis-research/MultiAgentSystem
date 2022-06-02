#IMPORTS
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, RNN, InputLayer, SimpleRNN
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver, dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import utils
from tf_agents.agents.dqn.dqn_agent import DqnAgent, DdqnAgent
from tf_agents.environments import py_environment
from tf_agents.networks.q_network import QNetwork
from tf_agents.networks import q_network
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer, tf_uniform_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import wrappers
from tf_agents.specs import tensor_spec
from tf_agents.specs import array_spec
from tf_agents.utils import common
from pickle import dump, load
import ast
from pymongo import MongoClient
from datetime import datetime, timedelta

##CLASSES
from classes.airConditioning import Airconditioning
from classes.storageBattery import StorageBattery
from classes.chargingStationEV import ChargingStation



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#Enviroment

class RealScenario(py_environment.PyEnvironment):
    
    def __init__(self, objective_curve, flexibility = 15.0):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=26, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, name='observation')
        #Current step
        self._state = 0
        self._episode_ended = False        

        #Set base_consumption
        self.objective_curve = objective_curve
        #Set flexibility 
        self.flexibility = flexibility        
        #Air conditioning of the room
        self.ac = Airconditioning()
        #Storage battery
        self.sb = StorageBattery()
        # EV Charging Station
        self.cs = ChargingStation()
        self.cs.carArrival(1)
        self.cs.carArrival(2)
        self.consumption = 0
        self.cumulative_consumption = 0
    
    def get_cumulative_consumption(self):
        return self.cumulative_consumption

    def get_consumption(self):
        return self.consumption

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec
    def current_time_step(self):
        """Returns the current `TimeStep`."""
        return self._current_time_step()

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        self.ac = Airconditioning()
        #Storage battery
        self.sb = StorageBattery()
        # EV Charging Station
        self.cs = ChargingStation()
        self.cs.carArrival(1)
        self.cs.carArrival(2)
        self.consumption = 0
        self.cumulative_consumption = 0
        return ts.restart(np.array([self.objective_curve[self._state]], dtype=np.int32))

    def _step(self, action):
        self.ac.turnOff()
        self.cs.stop()
        self.sb.stop()
        if self._episode_ended:
            self._reset()
         # Do nothing.
        if action == 0:

            c = self.ac.turnOff()
        #Small change temperature
        elif action == 1:
            c = self.ac.smallChange()
        #Big Change temperature 
        elif action == 2:
            c = self.ac.bigChange()
        #EV Charge
        elif action == 3:
            c = self.cs.charge(1) + self.ac.turnOff()
        #EV Discharge
        elif action == 4:
            c = self.cs.discharge(1) + self.ac.turnOff()
        #SB Charge12
        elif action == 5:
            c = self.sb.charge12() + self.ac.turnOff()
        #SB Charge24
        elif action == 6:
            c = self.sb.charge24() + self.ac.turnOff()
        #SB Charge36
        elif action == 7:
            c = self.sb.charge36() + self.ac.turnOff()
        #SB Discharge12
        elif action == 8:
            c = self.sb.discharge12() + self.ac.turnOff()
        #SB Discharge24
        elif action == 9:
            c = self.sb.discharge24() + self.ac.turnOff()
        #SB Discharge24
        elif action == 10:
            c = self.sb.discharge24() + self.ac.turnOff()
        #Small Change Temperature and EV Charge
        elif action == 11:
            c = self.ac.smallChange() + self.cs.charge(1)
        #Small Change Temperature and EV Discharge
        elif action == 12:
            c = self.ac.smallChange() + self.cs.discharge(1)
        #Small Change Temperature and SB Charge12
        elif action == 13:
            c = self.ac.smallChange() + self.sb.charge12()
        #Small Change Temperature and SB Charge24
        elif action == 14:
            c = self.ac.smallChange() + self.sb.charge24()
        #Small Change Temperature and SB Charge36
        elif action == 15:
            c = self.ac.smallChange() + self.sb.charge36()
        #Small Change Temperature and SB Discharge12
        elif action == 16:
            c = self.ac.smallChange() + self.sb.discharge12()
        #Small Change Temperature and SB Discharge24
        elif action == 17:
            c = self.ac.smallChange() + self.sb.discharge24()
        #Small Change Temperature and SB Discharge24
        elif action == 18:
            c = self.ac.smallChange() + self.sb.discharge24()
        #Big Change Temperature and EV Charge
        elif action == 19:
            c = self.ac.bigChange() + self.cs.charge(1)
        #Big Change Temperature and EV Discharge
        elif action == 20:
            c = self.ac.bigChange() + self.cs.discharge(1)
        #Big Change Temperature and SB Charge12
        elif action == 21:
            c = self.ac.bigChange() + self.sb.charge12()
        #Big Change Temperature and SB Charge24
        elif action == 22:
            c = self.ac.bigChange() + self.sb.charge24()
        #Big Change Temperature and SB Charge36
        elif action == 23:
            c = self.ac.bigChange() + self.sb.charge36()
        #Big Change Temperature and SB Discharge12
        elif action == 24:
            c = self.ac.bigChange() + self.sb.discharge12()
        #Big Change Temperature and SB Discharge24
        elif action == 25:
            c = self.ac.bigChange() + self.sb.discharge24()
        #Big Change Temperature and SB Discharge24
        elif action == 26:
            c = self.ac.bigChange() + self.sb.discharge24()
        #EV Charge and SB Charge12
        elif action == 27:
            c = self.cs.charge(1) + self.sb.charge12() + self.ac.turnOff()
        #EV Charge and SB Charge24
        elif action == 28:
            c = self.cs.charge(1) + self.sb.charge24() + self.ac.turnOff()
        #EV Charge and SB Charge36
        elif action == 29:
            c = self.cs.charge(1) + self.sb.charge36() + self.ac.turnOff()
        #EV Charge and SB Discharge12
        elif action == 30:
            c = self.cs.charge(1) + self.sb.discharge12() + self.ac.turnOff()
        #EV Charge and SB Discharge24
        elif action == 31:
            c = self.cs.charge(1) + self.sb.discharge24() + self.ac.turnOff()
        #EV Charge and SB Discharge24
        elif action == 32:
            c = self.cs.charge(1) + self.sb.discharge24() + self.ac.turnOff()
        #EV Discharge and SB Charge12
        elif action == 33:
            c = self.cs.discharge(1) + self.sb.charge12() + self.ac.turnOff()
        #EV Discharge and SB Charge24
        elif action == 34:
            c = self.cs.discharge(1) + self.sb.charge24() + self.ac.turnOff()
        #EV Discharge and SB Charge36
        elif action == 35:
            c = self.cs.discharge(1) + self.sb.charge36() + self.ac.turnOff()
        #EV Discharge and SB Discharge12
        elif action == 36:
            c = self.cs.discharge(1) + self.sb.discharge12() + self.ac.turnOff()
        #EV Discharge and SB Discharge24
        elif action == 37:
            c = self.cs.discharge(1) + self.sb.discharge24() + self.ac.turnOff()
        #EV Discharge and SB Discharge24
        elif action == 38:
            c = self.cs.discharge(1) + self.sb.discharge24() + self.ac.turnOff()
        #Small Change Temperature, EV Charge and SB Charge12
        elif action == 39:
            c = self.ac.smallChange() + self.cs.charge(1) + self.sb.charge12()
        #Small Change Temperature, EV Charge and SB Charge24
        elif action == 40:
            c = self.ac.smallChange() + self.cs.charge(1) + self.sb.charge24()
        #Small Change Temperature, EV Charge and SB Charge36
        elif action == 41:
            c = self.ac.smallChange() + self.cs.charge(1) + self.sb.charge36()
        #Small Change Temperature, EV Charge and SB Discharge12
        elif action == 42:
            c = self.ac.smallChange() + self.cs.charge(1) + self.sb.discharge12()
        #Small Change Temperature, EV Charge and SB Discharge24
        elif action == 43:
            c = self.ac.smallChange() + self.cs.charge(1) + self.sb.discharge24()
        #Small Change Temperature, EV Charge and SB Discharge24
        elif action == 44:
            c = self.ac.smallChange() + self.cs.charge(1) + self.sb.discharge24()
        #Big Change Temperature, EV Charge and SB Charge12
        elif action == 45:
            c = self.ac.bigChange() + self.cs.charge(1) + self.sb.charge12()
        #Big Change Temperature, EV Charge and SB Charge24
        elif action == 46:
            c = self.ac.bigChange() + self.cs.charge(1) + self.sb.charge24()
        #Big Change Temperature, EV Charge and SB Charge36
        elif action == 47:
            c = self.ac.bigChange() + self.cs.charge(1) + self.sb.charge36()
        #Big Change Temperature, EV Charge and SB Discharge12
        elif action == 48:
            c = self.ac.bigChange() + self.cs.charge(1) + self.sb.discharge12()
        #Big Change Temperature, EV Charge and SB Discharge24
        elif action == 49:
            c = self.ac.bigChange() + self.cs.charge(1) + self.sb.discharge24()
        #Big Change Temperature, EV Charge and SB Discharge24
        elif action == 50:
            c = self.ac.bigChange() + self.cs.charge(1) + self.sb.discharge24()
        val = self.objective_curve[self._state] + c
        self.consumption = C
        self.cumulative_consumption += val
            

        if abs(val) > self.objective_curve[self._state] * (self.flexibility)/100 or self._state == len(self.objective_curve)-1 :
            self._episode_ended = True 
            reward = 5*len(self.objective_curve) if self._state == len(self.objective_curve)-1 else -50*(len(self.objective_curve)-self._state)
            return ts.termination(np.array([self.objective_curve[self._state]], dtype=np.int32), reward)
        elif val == 0:
            reward = 1
        else:
            reward =  -abs(val)
        
        

        self._state += 1
        return ts.transition(np.array([self.objective_curve[self._state]], dtype=np.int32), reward=reward)

        
    def render(self, mode = 'human'):
        print("\n")
        #print("Current_step {}, Objective curve {},  real consumption {}, ended: {}".format(self._state, self.objective_curve[self._state], self.consumption, self._episode_ended))
        self.sb.render()
        self.ac.render()
        self.cs.render()

#GLOBAL VARS

num_iterations = int(input("Num iterations (Integer)(Recommended +1000): "))  # @param

initial_collect_steps = 10000  # @param
collect_steps_per_iteration = 1  # @param
replay_buffer_capacity = 10000  # @param

fc_layer_params = (100,)

batch_size = 128  # @param
learning_rate = 1e-3  # @param
log_interval = 1000  # @param

num_eval_episodes = 100  # @param
eval_interval = int(input("Evaluation interval (Integer)(Recommended 100): "))  # @param


#ADITIONAL FUNCTIONS

def compute_avg_return(environment, policy, num_episodes=10):
    
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]



def mas(curve, flexibility):
    env = RealScenario(curve, flexibility)
    train_py_env = wrappers.TimeLimit(RealScenario(curve, flexibility), duration=1000)
    eval_py_env = wrappers.TimeLimit(RealScenario(curve, flexibility), duration=1000)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v2.Variable(0)

    tf_agent = DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn = common.element_wise_squared_loss,
            train_step_counter=train_step_counter)



    tf_agent.initialize()
    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=replay_buffer_capacity)

    replay_observer = [replay_buffer.add_batch]

    train_metrics = [
                tf_metrics.EnvironmentSteps(),
                tf_metrics.AverageReturnMetric(),

    ]
    dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size, single_deterministic_pass=False,
    num_steps=2).prefetch(3)
    driver = dynamic_step_driver.DynamicStepDriver(
                train_env,
                collect_policy,
                observers=replay_observer + train_metrics,
        num_steps=1)
    print(compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes))

    tf_agent.train = common.function(tf_agent.train)
    tf_agent.train_step_counter.assign(0)
    action_values = ["Do nothing", "Small change temperature", "Big Change temperature", "EV Charge", "EV Discharge", "SB Charge12", "SB Charge24" ,"SB Charge36", "SB Discharge12" ,"SB Discharge24", "SB Discharge36", "Small Change Temperature and EV Charge", "Small Change Temperature and EV Discharge", "Small Change Temperature and SB Charge12", "Small Change Temperature and SB Charge24", "Small Change Temperature and SB Charge36", "Small Change Temperature and SB Discharge12", "Small Change Temperature and SB Discharge24", "Small Change Temperature and SB Discharge36", "Big Change Temperature and EV Charge", "Big Change Temperature and EV Discharge", "Big Change Temperature and SB Charge12", "Big Change Temperature and SB Charge24", "#Big Change Temperature and SB Charge36", "Big Change Temperature and SB Discharge12", "Big Change Temperature and SB Discharge24", "Big Change Temperature and SB Discharge36", "EV Charge and SB Charge12", "EV Charge and SB Charge24", "EV Charge and SB Charge36", "EV Charge and SB Discharge12", "EV Charge and SB Discharge24", "EV Charge and SB Discharge36", "EV Discharge and SB Charge12", "EV Discharge and SB Charge24", "EV Discharge and SB Charge36", "EV Discharge and SB Discharge12", "EV Discharge and SB Discharge24", "EV Discharge and SB Discharge36", "Small Change Temperature, EV Charge and SB Charge12", "Small Change Temperature, EV Charge and SB Charge24", "Small Change Temperature, EV Charge and SB Charge36", "EV Charge and SB Discharge12", "Small Change Temperature, EV Charge and SB Discharge24", "Small Change Temperature, EV Charge and SB Discharge36", "Big Change Temperature, EV Charge and SB Charge12", "Big Change Temperature, EV Charge and SB Charge24", "Big Change Temperature, EV Charge and SB Charge36", "Big Change Temperature, EV Charge and SB Discharge12", "Big Change Temperature, EV Charge and SB Discharge24",  "Big Change Temperature, EV Charge and SB Discharge36"]

    final_time_step, policy_state = driver.run()
    iterator = iter(dataset)
    for i in range(10000):
        final_time_step, _ = driver.run(final_time_step, policy_state)

    episode_len = []
    step_len = []
    for i in range(num_iterations):
        final_time_step, _ = driver.run(final_time_step, policy_state)
        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience=experience)
        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))
            step_len.append(step)

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            print('step = {0}: AVG REWARDS = {1}'.format(step, avg_return))


    action_list = []
    env.reset()
    cont = 0

    
    time_step = eval_env._reset()

    while not time_step.is_last():
        action_step = tf_agent.policy.action(time_step)
        action_list.append(tf.get_static_value(action_step.action[0]))
        time_step = eval_env.step(action_step.action)
        env.step(action_step.action)
        env.render()
        print("Current consumption: {}".format(curve[cont]))
        print("Consumption: {}".format(env.get_consumption()))
        print("Cumulative consumption: {}".format(env.get_cumulative_consumption()))
        print("Action done: {}".format(action_values[action_step.action[0]]))
        cont+=1
    train_env.close()
    eval_env.close()
    env.close()

    print()
    for a in action_list:
        print((action_values[a]))


def get_output():
    try:
        client = MongoClient(host = 'mongodb://192.168.48.206:27017/', username="cemosa", password="ebalance")

        db = client.Ebalance
        col = db["emv210"]

        data = col.find({},{"datetime" : 1, "kwh_p_tot_r" : 1}).sort("_id", -1).limit(1) 

        timestamp = pd.to_datetime(data[0]["datetime"])
        dow = timestamp.dayofweek
        timestamp = timestamp.strftime('%d/%m/%Y %H:%M:%S')
        day = timestamp[0:2]
        month = timestamp[3:5]
        year = timestamp[6:10]
        hour = timestamp[11:13]
        minute = timestamp[14:16]
        second = timestamp[17:19]
        
        
        col2 = db["openweather"]
        weather = 0
        data2 = col2.find().sort("_id", -1).limit(1)
        if data2[0]["weather"] == "Clouds":
            weather = 0
        elif data2[0]["weather"] == "Clear":
            weather = 1
        elif data2[0]["weather"] == "Rain":
            weather = 2
        elif data2[0]["weather"] == "Drizzle":
            weather = 3
        elif data2[0]["weather"] == "Mist":
            weather = 4
        elif data2[0]["weather"] == "Thunderstorm":
            weather = 5
        elif data2[0]["weather"] == "Haze":
            weather = 6
        elif data2[0]["weather"] == "Fog":
            weather = 7

        scaler = load(open("models/scaler.pkl", 'rb'))
        simplernn_model = tf.keras.models.load_model("models/model_simple_rnn.h5")  
        dense_model = tf.keras.models.load_model("models/model_dense_3.h5")
        lstm_model = tf.keras.models.load_model("models/model_lstm.h5")
        new_x = np.array([[47.0, 1, 26.09, 30.8, 1014, 40, 4.12, 2, 1, 6, 2022, 13, 00, 00, 44.0, 25.4, 23.7, 61.7]])

        X = scaler.transform(new_x)    
        dense_pred = dense_model.predict(X)
        lstm_pred = lstm_model.predict(X)
        simplernn_pred = simplernn_model.predict(X)
        curve = []
        for i, _ in enumerate(lstm_pred):
            curve.append((lstm_pred[i]+dense_pred[i]+simplernn_pred[i])/3)

        flexibility = input("Enter pattern consumption: ")
        print(curve[0].tolist())      
        mas(curve[0].tolist(), float(flexibility))  
    except Exception as ex:
        print(ex)
       

if __name__ == '__main__':
    get_output()
    
    
  
