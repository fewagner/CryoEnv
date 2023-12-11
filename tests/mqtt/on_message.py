import json
import numpy as np
# from IPython.display import display, clear_output
import os
import time

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
                    
            if data['TPA'] <= 0 or data['TPA'] >= 10.1:
                print('Ignoring message with TPA not in (0, 10.1).')
            elif time.time() - userdata['timer'] < 90:
                print('Ignoring message, initialising new episode for another {} sec.'.format(90 - time.time() + userdata['timer']))
            else:

                # calc state, reward - norm them
                if userdata['tpa_in_state']:
                    relaxation_factor = np.exp(-userdata['testpulse_interval']/userdata['tau'])
                    bias_normed = (2 * (data["BiasCurrent"] - userdata['Ib_range'][0]) / (userdata['Ib_range'][1] - userdata['Ib_range'][0]) - 1)
                    dac_normed = 2 * (data["DAC"] - userdata['dac_range'][0]) / (userdata['dac_range'][1] - userdata['dac_range'][0]) - 1
                    new_state = np.array([data["PulseHeight"] / userdata['adc_range'][1] * 2 - 1, 
                                          data["RMS"] / userdata['adc_range'][1] * 2 - 1,
                                          bias_normed,
                                          dac_normed,
                                          data["TPA"]/10 * 2 - 1,
                                          userdata['state'][5]*relaxation_factor - (1 - relaxation_factor)*bias_normed,
                                          userdata['state'][6]*relaxation_factor - (1 - relaxation_factor)*dac_normed
                                         ])
                else:
                    new_state = np.array([data["PulseHeight"] / userdata['adc_range'][1] * 2 - 1, 
                      data["RMS"] / userdata['adc_range'][1] * 2 - 1,
                      2 * (data["BiasCurrent"] - userdata['Ib_range'][0]) / (userdata['Ib_range'][1] - userdata['Ib_range'][0]) - 1,
                      2 * (data["DAC"] - userdata['dac_range'][0]) / (userdata['dac_range'][1] - userdata['dac_range'][0]) - 1,
                     ])

                rms = data['RMS']
                ph = np.maximum(data["PulseHeight"], rms/5)
                samples = np.array(data["Samples"]).flatten() / 65536. * 10.
                lrdiff = np.mean(samples[-50:]) - np.mean(samples[:50])
                penalty = userdata['penalty'] if lrdiff > ph/2 else 0.
                if userdata['log_reward']:
                    reward = - np.log(rms * data['TPA'] / ph*(1+userdata['ph_amp'])) - userdata['omega'] * np.sum((new_state[1:] - userdata['state'][1:]) ** 2) - penalty
                elif userdata['inv_reward']:
                    reward = ph * (1+userdata['ph_amp']) / rms / data['TPA'] /  - userdata['omega'] * np.sum((new_state[1:] - userdata['state'][1:]) ** 2) - penalty
                else:
                    reward = - rms * data['TPA'] / ph * (1+userdata['ph_amp']) - userdata['omega'] * np.sum((new_state[1:] - userdata['state'][1:]) ** 2) - penalty
                print('Reward: ', reward)
                terminated = False
                truncated = False

                # write to buffer
                userdata['buffer'].store_transition(state = userdata['state'], 
                                              action = userdata['action'],
                                              reward = reward, 
                                              next_state = new_state, 
                                              terminal = terminated)
                userdata['pulse_memory'][userdata['buffer'].buffer_total[0], :] = samples
                print('buffer total: ', userdata['buffer'].buffer_total)

                # update state
                userdata['state'] = new_state

                # get new action
                if userdata['sweep'] is not None:
                    len_sweep = len(userdata['sweep'])
                else:
                    len_sweep = 0
                if userdata['buffer'].buffer_total < len_sweep and not userdata['greedy']:
                    action = userdata['sweep'][userdata['buffer'].buffer_total[0]].reshape(1,-1)
                    print('Performing initial sweep, step nmbr {}/{}.'.format(userdata['buffer'].buffer_total[0]+1, len(userdata['sweep'])))
                elif userdata['buffer'].buffer_total < userdata['learning_starts'] and not userdata['greedy']:
                    action = userdata['env'].action_space.sample().reshape(1,-1)
                    print('Taking random action to fill buffer.')
                elif not os.path.isfile(userdata['path_models'] + 'policy.pt'):
                    action = userdata['env'].action_space.sample().reshape(1,-1)
                    print('Taking random action b/c no policy.pt file in {}.'.format(userdata['path_models']))
                elif userdata['buffer'].buffer_total % userdata['steps_per_episode'] == 0 and not userdata['greedy']:
                    action = userdata['env'].action_space.sample().reshape(1,-1)
                    action[0,np.random.randint(0,1)] = np.random.choice([-1, 1], size=1)
                    userdata['timer'] = time.time()
                    print('Taking reset action to initialise new episode.')
                else:
                    userdata['agent'] = SoftActorCritic.load(userdata['env'], userdata['path_models'])
                    action, _ = userdata['agent'].predict(userdata['state'], greedy=userdata['greedy']) 
                    greedy_action, greedy_likelihood = userdata['agent'].predict(userdata['state'], greedy=True)
                    print('greedy action is: {}, with likelihood: {}'.format(greedy_action, np.exp(greedy_likelihood)))

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
