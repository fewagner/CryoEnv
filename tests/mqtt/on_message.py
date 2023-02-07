import json
import numpy as np
# from IPython.display import display, clear_output
import os

from cryoenv.mqtt import SoftActorCritic, check


def receive_as_control(client, userdata, msg):
    
    try:
        if msg.topic == userdata['acknowledge_msg']['topic']:
            data = json.loads(msg.payload)
            userdata['action'] = np.array([data["DAC"], data["BiasCurrent"]])
            
            print('Acknowledge received and action set to {}'.format(userdata['action']))
            
            # norm
            userdata['action'][0] = 2 * (userdata['action'][0] - userdata['dac_range'][0]) / (userdata['dac_range'][1] - userdata['dac_range'][0]) - 1
            userdata['action'][1] = 2 * (userdata['action'][1] - userdata['Ib_range'][0]) / (userdata['Ib_range'][1] - userdata['Ib_range'][0]) - 1
            
        elif msg.topic == userdata['subscribe_acknowledge_msg']['topic']:
            data = json.loads(msg.payload)
            print('Subscription to channel {} acknowledged.'.format(data['SubscribeToChannel']))
            
        elif msg.topic == userdata['trigger_msg']['topic']:
            # clear_output(wait=True)
            os.system('clear')

            # get data
            data = json.loads(msg.payload)
            print('message received: ')
            for k,v in zip(data.keys(), data.values()):
                if k == 'Samples':
                    print(k,np.array(v).shape)
                else:
                    print(k,v)

            # calc state, reward - norm them
            new_state = np.array([data["PulseHeight"] / userdata['adc_range'][1] * 2 - 1, 
                                  data["RMS"] / userdata['adc_range'][1] * 2 - 1,
                                  2 * (data["BiasCurrent"] - userdata['Ib_range'][0]) / (userdata['Ib_range'][1] - userdata['Ib_range'][0]) - 1,
                                  2 * (data["DAC"] - userdata['dac_range'][0]) / (userdata['dac_range'][1] - userdata['dac_range'][0]) - 1,
                                 ])
            
            rms = data['RMS']
            ph = np.maximum(data["PulseHeight"], rms/5)
            reward = - rms * data['TPA'] / ph - userdata['omega'] * np.sum((new_state[1:] - userdata['state'][1:]) ** 2)
            print('Reward: ', reward)
            terminated = False
            truncated = False

            # write to buffer
            userdata['buffer'].store_transition(state = userdata['state'], 
                                          action = userdata['action'],
                                          reward = reward, 
                                          next_state = new_state, 
                                          terminal = terminated)
            userdata['pulse_memory'][userdata['buffer'].buffer_total[0], :] = np.array(data['Samples']).flatten()
            print('buffer total: ', userdata['buffer'].buffer_total)

            # update state
            userdata['state'] = new_state

            # get new action
            if userdata['buffer'].buffer_total > userdata['learning_starts']:
                userdata['agent'] = SoftActorCritic.load(userdata['env'], userdata['path_models'])
                action, _ = userdata['agent'].predict(userdata['state'], greedy=userdata['greedy']) 
                greedy_action, greedy_likelihood = userdata['agent'].predict(userdata['state'], greedy=True)
                print('greedy action is: {}, with likelihood: {}'.format(greedy_action, np.exp(greedy_likelihood)))
            else:
                action = userdata['env'].action_space.sample().reshape(1,-1)
                print('Taking random action.')

            # respond
            payload_response = {
                "ChannelID": userdata['channel'],
                "DAC": userdata['dac_range'][0] + (float(action[0,0]) + 1)/2*(userdata['dac_range'][1] - userdata['dac_range'][0]),
                "BiasCurrent": userdata['Ib_range'][0] + (float(action[0,1]) + 1)/2*(userdata['Ib_range'][1] - userdata['Ib_range'][0]),
            }

            # plot 
            print('message with greedy = {} respond: {}'.format(userdata['greedy'], payload_response))

            result = client.publish(userdata['set_pars_msg']['topic'], json.dumps(payload_response))
            check(result)
            
            if (userdata['greedy'] and userdata['buffer'].buffer_total[0] > userdata['inference_steps']) or (not userdata['greedy'] and userdata['buffer'].buffer_total[0] > userdata['env_steps']):
                client.disconnect()
            
        else:
            print('Message topic unknown: ', msg.topic)
        
    except KeyError as err_msg:
        print('KeyError: ', err_msg)
        pass
    

def receive_as_daq(client, userdata, msg):
    
    try:
        if msg.topic == userdata['subscribe_channel_msg']['topic']:
            
            data = json.loads(msg.payload)
            
            userdata['channel'] = data['SubscribeToChannel']
       
            payload_response = {
                "SubscribeToChannel": userdata['channel'],
            }

            result = client.publish(userdata['subscribe_acknowledge_msg']['topic'], json.dumps(payload_response))
            check(result)
            
        elif msg.topic == userdata['set_pars_msg']['topic']:
            
            data = json.loads(msg.payload)

            userdata['action'] = np.array([data["DAC"],  # comes not normed
                                           data["BiasCurrent"]])

            userdata['msg_received'] += 1

            payload_response = {
                "ChannelID": userdata['channel'],
                "DAC": float(userdata['action'][0]),
                "BiasCurrent": float(userdata['action'][1]),
            }
            
            # norm
            userdata['action'][0] = 2 * (userdata['action'][0] - userdata['dac_range'][0]) / (userdata['dac_range'][1] - userdata['dac_range'][0]) - 1
            userdata['action'][1] = 2 * (userdata['action'][1] - userdata['Ib_range'][0]) / (userdata['Ib_range'][1] - userdata['Ib_range'][0]) - 1

            result = client.publish(userdata['acknowledge_msg']['topic'], json.dumps(payload_response))
            check(result)
            
        else:
            print('Message topic unknown: ', msg.topic)
    
    except KeyError as err_msg:
        print('KeyError: ', err_msg)
        pass