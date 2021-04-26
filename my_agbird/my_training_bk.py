# encoding: utf-8

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
import sys
from collections import deque

from My_DQN import Agent


def main(cont):
    print('making environment')
    #env = gym.make('BaselineEnv-v0', log_level=logger.INFO, select_side='BLUE')
    env = gym.make('SelfplayEnv-v0', log_level=logger.INFO)
    print('making environment finished')
    EnvConfig.situation = 1
    EnvConfig.record = 1
    agent = Agent()
    if cont:
        print('cont is, ', cont)
        agent.get_buffer()
    try:
        # agent 1
        while True:
            os.system('rm -rf ~/plugin_exe/package/data/data*')
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
            current_n = []
            current_s = []
            rule1 = Rule(evals = True)
            rule2 = Rule(evals = False)

            if agent.buffer.is_ready():
                break

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
                    print('step: ',step , '  ______shoot distance:    ', rule1.R , '           missle num:    ', obs[0].sSMSData.nAAMMissileNum ,  '       is continuous shoot?:', obs[0].nMslPropertyNum)
                    n = step * 0.2
                    current_n.append(n)
                    current_s.append(s)
                ############################################################ collect data end #################################################################################

                actions = [ac1, ac2]
                #actions = ac1
                obs, reward, done, info = env.step(actions)
                #print("\rstep: {}                   ".format(env.step_num), end='')
                if done: #obs[0].sOtherInfo.bOver or obs[1].sOtherInfo.bOver:
                    #print('step done')
                    #print(obs[0].sOtherInfo.nEndReason, obs[1].sOtherInfo.nEndReason)
                    #print('step done test')
                    #print(obs[0].sOtherInfo.nEndReason, obs[1].sOtherInfo.nEndReason)
                    total_time = step / 5
                    print(agent.buffer.count)
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
                        for j in range(len(current_n)):
                            agent.buffer.add_sample(current_s[j], r, total_time - current_n[j])
                            if agent.buffer.is_ready():
                                print('start training!!!!!!!!')
                                agent.pre_data()
                                agent.train()  
                                agent.save_model(0) 
                                break    
                        agent.store_buffer()         
                    break
                step += 1
    except Exception:
        print("got exception:", traceback.format_exc())
        env.close()
        exit(1)
    finally:
        env.close()
        exit(0)
        


def cal_enemy_heading(obs, enemy_state):
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

            # elif self.turn_step == 2:
            #     print('defend third step...............................................')
            #     act.siPlaneControl.iCmdID = 19
            #     act.siPlaneControl.fCmdNy = 6  # guo zai
            #     act.siPlaneControl.fCmdPitchDeg = 0  # pitch
            #     act.siPlaneControl.fCmdPhi = 60
            #     act.siPlaneControl.fCmdRollDeg = 60
            #     act.siPlaneControl.iTurnDirection = 1
            #     act.siPlaneControl.fCmdBdyPsiDeg = self.first_heading
            #     act.siPlaneControl.fCmdHeadingDeg = self.first_heading


if __name__ == '__main__':
    cont = int(sys.argv[1])
    main(cont)
