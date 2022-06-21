from psiturk.db import db_session,init_db
from psiturk.models import Participant
import os
import string
import itertools

from datetime import timedelta
import random

from urllib.request import urlopen
import unittest

from psiturk.psiturk_shell import PsiturkNetworkShell
from psiturk.psiturk_config import PsiturkConfig
import psiturk.experiment_server_controller as control


class PsiturkForTimeTests:
    def __init__(self):
        #subject params
        self.times = [0,.5,1,2] #times to test
        self.statuses = [1,2,3] #started, got past 'complete instructions' call, completed
        self.repeat_error = [True, True, True, True, True, True, False, False, False, False, False, False]
        
        self.participant_settings = self.construct_participants()
        self.shell = None
    
    def construct_participants(self):
        def create_mturk_tuple(idx):
            hitid = (string.ascii_uppercase[idx]+str(idx))*15
            workerid = (str(idx)+string.ascii_uppercase[idx])*7
            assignmentid = string.ascii_uppercase[idx]*30
            return (workerid, assignmentid, hitid)
        
        # create test subjects 


        participant_settings = [[*create_mturk_tuple(prod_num),*prod] for prod_num, prod in \
                                enumerate(itertools.product(self.times,self.statuses))]
        
        return participant_settings
   
    def add_participants_to_db(self):
        def add_participant(worker_id, assignment_id, hit_id, time_offset=0,  status = 1, mode="live"):
            # default settings that don't matter
            subj_cond = 1
            subj_counter = 0
            worker_ip = "UNKNOWN"
            language = "UNKNOWN"
            browser = "UNKNOWN"
            platform = "UNKNOWN"

            participant_attributes = dict(
                    assignmentid=assignment_id,
                    workerid=worker_id,
                    hitid=hit_id,
                    cond=subj_cond,
                    counterbalance=subj_counter,
                    ipaddress=worker_ip,
                    browser=browser,
                    platform=platform,
                    language=language,
                    mode=mode
                )

            part = Participant(**participant_attributes)

            part.status = status

            fake_experiment_duration = random.randrange(25,35)
            part.beginhit = part.beginhit - timedelta(days=time_offset*365)
            part.beginexp = part.beginhit
            part.endhit = part.beginhit + timedelta(minutes=fake_experiment_duration)

            db_session.add(part)
            db_session.commit()
            
        #add participant with inner add_participant function
        for part_info in self.participant_settings:
            add_participant(*part_info) #add participant to database

    def setup_db(self):
        self.teardown_db()
        init_db()
        
    def teardown_db(self):
        if os.path.exists("participants.db"):
            os.remove("participants.db")
            
    def turn_psiturk_server_on(self):
        config = PsiturkConfig()
        config.load_config()
        server = control.ExperimentServerController(config)
        self.shell = PsiturkNetworkShell(config, server, sandbox = True)
        self.shell.server_on()
        
    def turn_psiturk_server_off(self):
        if self.shell:
            self.shell.server_off()
    
    def full_setup(self):
        #turn off psiturk shell if needed
        self.turn_psiturk_server_off()
        #tear down old db if needed
        self.teardown_db()
        
        #set up db and add participants
        self.setup_db()
        self.add_participants_to_db()
        
        #turn psiturk shell on
        self.turn_psiturk_server_on()
        
    def full_teardown(self):
        #turn psiturk shell off
        self.turn_psiturk_server_off()
        
        #tear down db
        self.teardown_db()
       
    
class TimeTest(unittest.TestCase):
    def setUp(self):
        self.psiturk_running = PsiturkForTimeTests()
        self.psiturk_running.full_setup()
    def tearDown(self):
        self.psiturk_running.full_teardown()

    def test_repeats(self):
        for part_idx, part_info in enumerate(self.psiturk_running.participant_settings):
            url = "http://0.0.0.0:22369/turkprime?workerId={}&assignmentId={}&hitId={}&yearLimit=1&mode=live".format(part_info[0], "NEW_ASSIGNMENT", "NEW_HIT")

            response = urlopen(url)
            html = str(response.read())

            assert response.getcode() == 200

            if self.psiturk_running.repeat_error[part_idx]:
                assert '1010' in html #should be an unallowed repeat
            else:
                assert '1010' not in html #they are allowed to repeat

        
