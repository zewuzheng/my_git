## My rule designed to train shooting agent ###############

import angrybirds
import gym
import traceback
from angrybirds.lib import logger
from angrybirds.plugins.game.firecontroller.fc import FireControl
from angrybirds.config.plane_config import RedPlane
from angrybirds.config.plane_config import BluePlane
from angrybirds.config.plane_config import EnvConfig
import numpy as np
import random
import os
import time
import sys
from collections import deque
import ray
from collections import deque

import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



def main(cont):

    ############ start ray #####################################################
    num_env_parallels = 20
    ray_host = os.environ.get("RAY_HEAD_SERVICE_HOST", "39.99.43.183")
    ray_port = os.environ.get("RAY_HEAD_SERVICE_PORT", 6379)
    node_ip_address = os.environ.get("NODE_IP_ADDRESS", None)
    ray_config = dict(address=f"{ray_host}:{ray_port}", node_ip_address=node_ip_address)
    print(f"Starting ray on host {ray_host}")
    ray.init(**ray_config)
    print("Started ray")

    ############ check if pods ready ###########################################
    for trial in range(30):
        ready_num = ray.available_resources()['worker_node']
        print(f'Ready workers: {ready_num}')
        if ready_num >= num_env_parallels:
            break
        time.sleep(1.0)
        if trial == 30 - 1:
            exit(0)

    agent = Agent()
    if cont:
        print('cont is, ', cont)
        agent.get_buffer()
    
    obj_ref = {}
    print('going into parallel collection......')
    actors = [collect_d.remote() for _ in range(num_env_parallels)]
    idle_actors = deque(actors)
    ready = False
    while True:
        while len(idle_actors) > 0:
            actor = idle_actors.popleft()
            future = actor.collect_data.remote()
            obj_ref[future] = actor

        ready_idx, _ = ray.wait(list(obj_ref.keys()), len(obj_ref.keys()), timeout=5.0)
        
        for ready_ind in ready_idx:
            result = ray.get(ready_ind)
            idle_actors.append(obj_ref[ready_ind])
            obj_ref.pop(ready_ind)    

            if result != 0:
                print(agent.buffer.count)
                for s, r, n in zip(result[0], result[1], result[2]):
                    agent.buffer.add_sample(s, r, n)
                    if agent.buffer.is_ready():
                        ready = True
                        break
            if ready:
                break

        agent.store_buffer()
        if ready:
            break
        
    agent.pre_data()
    agent.train()
    agent.save_model(21426) 
      

        

@ray.remote
class collect_d():
    @staticmethod
    def collect_data():
        env = gym.make('SelfplayEnv-v0', log_level=logger.INFO)
        EnvConfig.situation = 1
        EnvConfig.record = 1
        jushu = 5
        
        try:
            _s = []
            _r = []
            _n = []
            for _ in range(jushu):
                os.system('rm -rf /work/plugin_exe/package/data/data*')
                RedPlane.longitude = 115
                RedPlane.latitude = 30.17
                RedPlane.height = 7000
                RedPlane.heading = 0
                RedPlane.speed = 300.0
                RedPlane.fuel = 4000.0

                # agent 2
                BluePlane.longitude = 115
                BluePlane.latitude = 31.1
                BluePlane.height = 7000
                BluePlane.heading = 180
                BluePlane.speed = 300.0
                BluePlane.fuel = 4000.0
                obs, reward, done, info = env.reset()
                step = 0
                rule1 = Rule(evals = True)
                rule2 = Rule(evals = False)
                current_n = []
                current_s = []

                while True:
                    fc = FireControl()
                    ac1 = env.coder.generate_aiiputdata()
                    ac2 = env.coder.generate_aiiputdata()
                    ac1 = fc.execute(1, ac1)
                    ac2 = fc.execute(2, ac2)

                    ########################################################### rule begin ######################################################################################
                    rule1.my_rule(obs[0], ac1)
                    rule2.my_rule(obs[1], ac2)
                    ########################################################### rule end #########################################################################################
                    ########################################################### collect data begin ###############################################################################
                    
                    if step >= 10 and rule1.cmd_shoot:
                        Msl_sky = -2 if not obs[0].nMslPropertyNum else 2
                        s = [rule1.enemy_state['ve'] / 1000,rule1.enemy_state['vn'] / 1000,rule1.enemy_state['vu'] / 1000,
                            rule2.enemy_state['ve'] / 1000,rule2.enemy_state['vn'] / 1000,rule2.enemy_state['vu'] / 1000,
                            rule1.R / 8 - 6, obs[0].sSMSData.nAAMMissileNum - 2.5, Msl_sky]
                        n = step * 0.2
                        current_n.append(n)
                        current_s.append(s)

                    ############################################################ collect data end #################################################################################

                    actions = [ac1, ac2]
                    obs, reward, done, info = env.step(actions)
                    if done:
                        print('done :', step)
                        total_time = step / 5
                        if obs[0].sOtherInfo.nEndReason == 3 and obs[1].sOtherInfo.nEndReason == 3:
                            r = 1
                        elif obs[0].sOtherInfo.nEndReason == 3:
                            r = -10
                        elif obs[1].sOtherInfo.nEndReason == 3:
                            r = 5
                        elif obs[0].sOtherInfo.nEndReason == 5 and obs[1].sOtherInfo.nEndReason == 5:
                            r = -2
                        elif obs[0].sOtherInfo.nEndReason == 5:
                            r = -20
                        else:
                            r = 0

                        if len(current_n) > 0:
                            print('have been shot!')
                            _s += current_s
                            _r += [r] * len(current_n)
                            _n += current_n
                        else:
                            print('no shoot!')
                        break
                    step += 1
            env.close()
            time.sleep(5.0)
            return [_s, _r, _n]
        except Exception:
            print("got exception:", traceback.format_exc())
            if len(_n) > 0:
                return [_s, _r, _n]
            else:
                return 0
        finally:
            env.close()


def cal_enemy_heading(obs, enemy_state):
    """检查敌机角度
    """
    DLs = [i for i in range(obs.AATargetDataListNum) if obs.sAATargetData[i].bDLTarget]
    AEs = [obs.sAATargetData[i].fTargetDistanceAE_m for i in range(obs.AATargetDataListNum) if
           obs.sAATargetData[i].bAETarget]
    if AEs or DLs:
        east_b, north_b, up_b = \
            enemy_state['east'], enemy_state['north'], enemy_state['up']
        # 我方位置
        east_r, north_r = gps2xy(obs.sFighterPara.dLongtitude_rad, obs.sFighterPara.dLatitude_rad)
        up_r = obs.sFighterPara.fAltitude_m / 1e3  # km
        delta_east, delta_north = east_b - east_r, north_b - north_r
        heading = np.rad2deg(np.arctan2(delta_east, delta_north))
    else:
        heading = 180
    return heading


def cal_relative_dis(obs, enemy_state):
    """检查两机相对距离
    """
    DLs = [i for i in range(obs.AATargetDataListNum) if obs.sAATargetData[i].bDLTarget]
    AEs = [obs.sAATargetData[i].fTargetDistanceAE_m for i in range(obs.AATargetDataListNum) if
           obs.sAATargetData[i].bAETarget]
    if AEs:
        R = min(AEs) / 1e3
    elif isinstance(enemy_state, dict):
        east_b, north_b, up_b = \
            enemy_state['east'], enemy_state['north'], enemy_state['up']
        # 我方位置
        east_r, north_r = gps2xy(obs.sFighterPara.dLongtitude_rad, obs.sFighterPara.dLatitude_rad)
        up_r = obs.sFighterPara.fAltitude_m / 1e3  # km
        delta_east, delta_north, delta_up = east_b - east_r, north_b - north_r, up_b - up_r
        # 特征计算
        R = np.linalg.norm([delta_east, delta_north, delta_up])  # 相对距离km
        # 雷达量程控制和目标的距离相关，留出来了40%的余量
    elif DLs:
        data = obs.sAATargetData[DLs[0]]
        east_b, north_b = gps2xy(data.fLonDL_rad, data.fLatDL_rad)
        up_b = data.nAltDL_m / 1e3
        east_r, north_r = gps2xy(obs.sFighterPara.dLongtitude_rad, obs.sFighterPara.dLatitude_rad)
        up_r = obs.sFighterPara.fAltitude_m / 1e3  # km
        R = np.sqrt((north_b - north_r) ** 2 + (east_b - east_r) ** 2 + (up_b - up_r) ** 2)
    else:
        R = 1e6  # 无限大
    return R, True if AEs else False


def _in_range(angle):
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360
    return angle


def gps2xy(longtitude, latitude):
    east = (longtitude - np.deg2rad(123.4)) * np.cos(np.deg2rad(42.2)) * 6371
    north = (latitude - np.deg2rad(42.2)) * 6371
    return east, north


def get_x_y_z(distance, pitch, yaw):
    up = distance * np.sin(pitch)
    east = distance * np.cos(pitch) * np.sin(yaw)
    north = distance * np.cos(pitch) * np.cos(yaw)
    return east, north, up


def get_index_from_data(obs):
    """
    获取AATarget和TargetInList中的锁定目标的下标
    """
    AATargetLots = {obs.sAATargetData[i].nAllyLot for i in range(obs.AATargetDataListNum) if
                    obs.sAATargetData[i].bAETarget}
    TargetInLots = {obs.sAttackList.sTargetInListData[i].nLot for i in range(obs.sAttackList.nTargetInListNum)}
    CommonLots = set.intersection(AATargetLots, TargetInLots)

    if len(CommonLots) == 1:  # 如果共有的Lot只有一个就直接返回
        AATargetIndex = [i for i in range(obs.AATargetDataListNum) if
                         obs.sAATargetData[i].bAETarget and obs.sAATargetData[i].nAllyLot in CommonLots][0]
        TargetInIndex = \
        [i for i in range(obs.sAttackList.nTargetInListNum) if obs.sAttackList.sTargetInListData[i].nLot in CommonLots][
            0]
        return AATargetIndex, TargetInIndex
    elif len(CommonLots) == 0:  # 如果共有的Lot一个都没有
        if len(AATargetLots) == 1 and len(TargetInLots) == 1:  # 检查一下是不是都只有一个目标，如果是直接返回
            AATargetIndex = [i for i in range(obs.AATargetDataListNum) if
                             obs.sAATargetData[i].bAETarget and obs.sAATargetData[i].nAllyLot in AATargetLots][0]
            TargetInIndex = [i for i in range(obs.sAttackList.nTargetInListNum) if
                             obs.sAttackList.sTargetInListData[i].nLot in TargetInLots][0]
            return AATargetIndex, TargetInIndex
        else:
            return 0, 0  # TODO 否则先随便返回一个
    else:  # 如果有不止一个，再用规则判断

        def parse_data(data):
            dis = data.fTargetDistanceAE_m / 1e3  # km
            theta = data.fTargetPitchAE_rad
            psi = data.fTargetAzimAE_rad + obs.sFighterPara.fHeading_rad
            x, y, z = get_x_y_z(dis, theta, psi)
            return x, y, z

        AEInds = [i for i in range(obs.AATargetDataListNum) if
                  obs.sAATargetData[i].bAETarget and obs.sAATargetData[i].nAllyLot in CommonLots]
        xyzs = np.array([parse_data(obs.sAATargetData[i]) for i in AEInds])
        index = indexFrom3dData(xyzs)  # 根据转换三维数据经PCA降维锁定目标
        AATargetIndex = AEInds[index]
        TargetInIndex = [i for i in range(obs.sAttackList.nTargetInListNum) if
                         obs.sAttackList.sTargetInListData[i].nLot == obs.sAATargetData[AATargetIndex].nAllyLot][0]
        return AATargetIndex, TargetInIndex


class Rule():
    def __init__(self, evals):
        self.eval = evals
        self.first_ind = 0
        self.first_heading = 0
        self.turn_step = 0
        self.straight_timer = 0
        self.missile_l = [0] * 20
        self.enemy_state = None
        self.old_shoot = deque([0] * 20)
        self.cmd_shoot = False
        self.R = 0
        if self.eval:
            self.shoot1 = random.uniform(35, 80)
            self.shoot2 = random.uniform(20, self.shoot1 + 10)
            self.shoot3 = random.uniform(15, self.shoot2 + 10)
            self.shoot4 = random.uniform(10, self.shoot3 + 10)
        else:
            self.shoot1 = random.uniform(50, 80)
            self.shoot2 = random.uniform(30, self.shoot1 + 20)
            self.shoot3 = random.uniform(25, self.shoot2 + 20)
            self.shoot4 = random.uniform(15, self.shoot3 + 20)

    def update_enemy_state(self, obs, beats):
        """
        更新敌机信息
        """
        def safe_update(src_dict, target_dict):
            if src_dict is None:
                src_dict = {}
            for k in target_dict:
                if not np.isnan(target_dict[k]):
                    src_dict[k] = target_dict[k]
                else:
                    src_dict[k] = src_dict.get(k, 0)
            return src_dict

        DLs = [
            i for i in range(obs.AATargetDataListNum)
            if obs.sAATargetData[i].bDLTarget
        ]
        if obs.sAttackList.nTargetInListNum:  # 如果雷达有信息优先用雷达信息
            aa_index, tl_index = get_index_from_data(obs)
            data = obs.sAATargetData[aa_index]
            e_r, n_r = gps2xy(obs.sFighterPara.dLongtitude_rad, obs.sFighterPara.dLatitude_rad)
            u_r = obs.sFighterPara.fAltitude_m / 1e3
            heading_r = obs.sFighterPara.fHeading_rad
            dis = data.fTargetDistanceAE_m / 1e3
            theta = data.fTargetPitchAE_rad
            psi = data.fTargetAzimAE_rad + heading_r
            delta_e, delta_n, delta_u = get_x_y_z(dis, theta, psi)
            e_b, n_b, u_b = e_r + delta_e, n_r + delta_n, u_r + delta_u
            ve_b = data.fVeAE_ms * 3.6
            vn_b = data.fVnAE_ms * 3.6
            vu_b = data.fVuAE_ms * 3.6
            self.enemy_state = safe_update(self.enemy_state, {
                "east": e_b,
                "north": n_b,
                "up": u_b,
                "ve": ve_b,
                "vn": vn_b,
                "vu": vu_b
            })
            fcm_data = obs.sAttackList.sTargetInListData[tl_index]
        elif DLs:  # 其次考虑数据链信息
            data = obs.sAATargetData[DLs[0]]
            e_b, n_b = gps2xy(data.fLonDL_rad, data.fLatDL_rad)
            u_b = data.nAltDL_m / 1e3
            ve_b = data.fVeDL_kmh
            vn_b = data.fVnDL_kmh
            vu_b = data.fVuDL_kmh
            self.enemy_state = safe_update(self.enemy_state, {
                "east": e_b,
                "north": n_b,
                "up": u_b,
                "ve": ve_b,
                "vn": vn_b,
                "vu": vu_b
            })
        else:
            if self.enemy_state is None:
                print('No enemy state!!!')


    def shoot_action(self, obs, act, en_heading):
        ########################### public calculation #################################################
        aa_index, tl_index = get_index_from_data(obs)
        act.sSOCtrl.nNTSIdAssigned = obs.sAttackList.sTargetInListData[tl_index].nLot
        act.sSOCtrl.bNTSAssigned = 1
        if (abs(_in_range(en_heading - np.rad2deg(obs.sFighterPara.fHeading_rad))) < 25 and not self.eval and not obs.nMslPropertyNum):
            visual = True
        elif self.eval:
            visual = True
        else:
            visual = False

        if visual and obs.sAttackList.nTargetInListNum and obs.sSMSData.nAAMMissileNum == 4 and self.R < self.shoot1:
            self.cmd_shoot = True
        elif visual and obs.sAttackList.nTargetInListNum and obs.sSMSData.nAAMMissileNum == 3 and self.R < self.shoot2:
            self.cmd_shoot = True
        elif visual and obs.sAttackList.nTargetInListNum and obs.sSMSData.nAAMMissileNum == 2 and self.R < self.shoot3:
            self.cmd_shoot = True
        elif visual and obs.sAttackList.nTargetInListNum and obs.sSMSData.nAAMMissileNum == 1 and self.R < self.shoot4:
            self.cmd_shoot = True
        else:
            self.cmd_shoot = False

        if True in self.old_shoot:
            self.cmd_shoot = False

        self.old_shoot.popleft()
        self.old_shoot.append(self.cmd_shoot)
        act.sOtherControl.bLaunch = self.cmd_shoot

    def check_msl_nearing(self, obs):
        """检查导弹告警信息
        """
        WarnMsl = [obs.sOtherInfo.sAlarm[i].bWarnMslNearing for i in range(obs.sOtherInfo.nAlarmNum)]
        if obs.sStateData.bWarnMslNearing is True:
            self.missile_l.append(1)
        else:
            self.missile_l.append(0)

        msl_first_come_flag = False
        msl_flag = False

        if len(self.missile_l) > 20:
            self.missile_l.pop(0)
            # # 0 0 1 则为True，第一次出现导弹告警      
            msl_first_come_flag = self.missile_l[-1] == 1 and max(self.missile_l[-12:-2]) < 1
            msl_flag = obs.sStateData.bWarnMslNearing is True or True in WarnMsl

            if msl_flag == 0:
                if max(self.missile_l[-11:-2]) < 1:
                    return False  # 真的很多0，才认为告警消失
                else:
                    return True
        return True if (msl_flag or msl_first_come_flag) else False

    def my_rule(self, obs, act):
        self.update_enemy_state(obs, 5)
        relative_dis, aes_flag = cal_relative_dis(obs, self.enemy_state)
        self.R = relative_dis
        en_heading = cal_enemy_heading(obs, self.enemy_state)
        msl_nearing = self.check_msl_nearing(obs)

        ######################################### shoot action #########################################################################################################
        self.shoot_action(obs, act, en_heading)
        ######################################### public command #######################################################################################################
        act.siPlaneControl.iVelType = 0  # speed type
        act.siPlaneControl.fLimitAlt = 2000  # altitude limit
        act.siPlaneControl.fThrustLimit = 120  # thrust limit
        act.siPlaneControl.fCmdSpd = 1.3  # speed
        ########################################## A missile is comming, escape from it #################################################################################
        if not msl_nearing:
            self.first_ind = 0
            ########################################## climbing to a safe height ############################################################################################
            if obs.sFighterPara.fAltitude_m < 4000:
                #print('climbing to safe height !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                act.siPlaneControl.iCmdID = 4
                act.siPlaneControl.fCmdNy = 2  # guo zai
                act.siPlaneControl.fCmdPitchDeg = 12  # pitch
                act.siPlaneControl.fCmdPhi = 0
                act.siPlaneControl.fCmdRollDeg = 0
                act.siPlaneControl.iTurnDirection = 0
                act.siPlaneControl.fCmdBdyPsiDeg = np.rad2deg(obs.sFighterPara.fHeading_rad)
                act.siPlaneControl.fCmdHeadingDeg = np.rad2deg(obs.sFighterPara.fHeading_rad)
            ########################################## turn to the enemy ####################################################################################################
            elif abs(_in_range(en_heading - np.rad2deg(obs.sFighterPara.fHeading_rad))) > 10:
                #print('turning to enemy ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                act.siPlaneControl.iCmdID = 12
                act.siPlaneControl.fCmdThrust = 120
                act.siPlaneControl.fCmdNy = 5  # guo zai
                act.siPlaneControl.fCmdPitchDeg = 0  # pitch
                act.siPlaneControl.fCmdPhi = 60
                act.siPlaneControl.fCmdRollDeg = 60
                act.siPlaneControl.iTurnDirection = 0
                act.siPlaneControl.fCmdBdyPsiDeg = en_heading
                act.siPlaneControl.fCmdHeadingDeg = en_heading
            ########################################## fly towards the enemy ################################################################################################
            else:
                #print('going forward ___________________________________________________________')
                act.siPlaneControl.iCmdID = 1
                act.siPlaneControl.fCmdNy = 1  # guo zai
                act.siPlaneControl.fCmdPitchDeg = 0  # pitch
                act.siPlaneControl.fCmdPhi = 0
                act.siPlaneControl.fCmdRollDeg = 0
                act.siPlaneControl.iTurnDirection = 0
                act.siPlaneControl.fCmdBdyPsiDeg = np.rad2deg(obs.sFighterPara.fHeading_rad)
                act.siPlaneControl.fCmdHeadingDeg = np.rad2deg(obs.sFighterPara.fHeading_rad)
        else:
            if self.first_ind == 0:
                self.first_heading = np.rad2deg(obs.sFighterPara.fHeading_rad)
                self.turn_step = 0
                self.straight_timer = 0
                self.first_ind = 1

            if abs(_in_range(np.rad2deg(obs.sFighterPara.fHeading_rad) - _in_range(
                    self.first_heading + 170))) <= 5 and self.turn_step == 0:
                self.turn_step = 1
            # elif self.straight_timer > 25 and self.turn_step == 1:
            #     self.turn_step = 2       

            if self.turn_step == 0:
                #print('defend first step...............................................')
                if self.R < 25:
                    act.siPlaneControl.fCmdNy = 5.7  # guo zai
                else:
                    act.siPlaneControl.fCmdNy = 5  # guo zai
                act.siPlaneControl.iCmdID = 13                  
                act.siPlaneControl.fCmdPitchDeg = 0  # pitch
                act.siPlaneControl.fCmdPhi = 75
                act.siPlaneControl.fCmdThrust = 120
                act.siPlaneControl.fCmdRollDeg = 7
                act.siPlaneControl.iTurnDirection = 1
                act.siPlaneControl.fCmdBdyPsiDeg = _in_range(self.first_heading + 120)
                act.siPlaneControl.fCmdHeadingDeg = self.first_heading
            elif self.turn_step == 1:
                #print('defend second step...............................................')
                act.siPlaneControl.iCmdID = 10
                act.siPlaneControl.fCmdNy = 4.5  # guo zai
                act.siPlaneControl.fCmdPitchDeg = 0  # pitch
                act.siPlaneControl.fCmdPhi = 0
                act.siPlaneControl.fCmdThrust = 120
                act.siPlaneControl.fCmdRollDeg = 7
                act.siPlaneControl.iTurnDirection = 1
                act.siPlaneControl.fCmdBdyPsiDeg = np.rad2deg(obs.sFighterPara.fHeading_rad)
                act.siPlaneControl.fCmdHeadingDeg = np.rad2deg(obs.sFighterPara.fHeading_rad)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ) 

    def Forward(self, state):
        return self.fc(state)


class Replay_buffer():
    def __init__(self, state_size):
        self.count = 0
        self.size = 300000
        self.data_type = np.dtype([('s', np.float64, state_size), 
                                   ('r', np.float64), ('n', np.float64)])
        self.buffer = [] #np.empty(0, dtype=self.data_type)

    def add_sample(self, s, r, n):
        if self.count == 0:
            self.buffer = []
        self.buffer.append((s, r, n))
        self.count += 1

    def is_ready(self):
        if self.count == self.size:
            self.buffer = np.array(self.buffer, dtype=self.data_type)
            return True
        else:
            return False


class Agent():
    def __init__(self):
        self.net= Net().float()
        self.buffer = Replay_buffer(9)
        self.optimizer = opt.Adam(self.net.parameters(), lr=1e-3)
        self.gamma = 0.995

    @torch.no_grad()
    def get_act(self, state):
        state = torch.from_numpy(state).float()
        q = self.net.Forward(state)
        return 0 if q <= 0 else 1

    def get_value(self, state):
     #   state = torch.from_numpy(state).float()
        q = self.net.Forward(state)
        return q 

    def train(self):
        for i in range(self.buffer.buffer.size // 15):
            print(f'iteration {i}...................................................')
            sample_index = np.random.choice(self.buffer.buffer.size,256)
            s = torch.tensor(self.buffer.buffer['s'], dtype=torch.float)[sample_index]
            r = torch.tensor(self.buffer.buffer['r'], dtype=torch.float).view(-1, 1)[sample_index]   ## n * 1
            n = torch.tensor(self.buffer.buffer['n'], dtype=torch.float).view(-1, 1)[sample_index]
            print(r)
            q_eval = self.get_value(s)
            loss = F.mse_loss(q_eval, (self.gamma ** n) * r)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save_model(self, t):
        torch.save(self.net.state_dict(), f'models/agent_{t}.pt')

    def load_model(self, path):
        self.net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def store_buffer(self):
        data = np.array(self.buffer.buffer)
        np.save('models/buffer.npy', data)
    
    def get_buffer(self):
        data = np.load('models/buffer.npy', allow_pickle=True)
        temp_buffer = data.tolist()
        self.buffer.buffer = [tuple(x) for x in temp_buffer]
        self.buffer.count = len(self.buffer.buffer)

    def pre_data(self):
        buf = np.array(self.buffer.buffer, dtype = self.buffer.data_type) if evals == True else self.buffer.buffer
        r_buf = buf['r']
        size_r = len(r_buf)
        r_ind = np.where(r_buf>0)[0]
        ratio = int((size_r // len(r_ind)) * 0.7)
        print('ratio is ', ratio / 0.7)
        for ind in r_ind:
            s1 = np.array([buf[ind] for _ in range(ratio)] , dtype = self.buffer.data_type)
            buf = np.concatenate((buf,s1), axis = 0)
        np.random.shuffle(buf)
        self.buffer.buffer = buf
        buf = []


if __name__ == '__main__':
    cont = int(sys.argv[1])
    main(cont)
