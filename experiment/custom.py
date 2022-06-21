# this file imports custom routes into the experiment server

from flask import Blueprint, Response, abort, current_app, request, jsonify, render_template, render_template_string
from traceback import format_exc

from psiturk.psiturk_config import PsiturkConfig
from psiturk.user_utils import PsiTurkAuthorization, nocache
from psiturk.experiment_errors import ExperimentError

import json
from datetime import timedelta, date, datetime
from sqlalchemy import exc, func

import re

# Database setup
from psiturk.db import db_session
from psiturk.models import Participant

import user_agents
# load the configuration options
config = PsiturkConfig()
config.load_config()

# if you want to add a password protect route use this
myauth = PsiTurkAuthorization(config)

# explore the Blueprint
custom_code = Blueprint(
    'custom_code', __name__,
    template_folder='templates',
    static_folder='static')

# Status codes
NOT_ACCEPTED = 0
ALLOCATED = 1
STARTED = 2
COMPLETED = 3
SUBMITTED = 4
CREDITED = 5
QUITEARLY = 6
BONUSED = 7
BAD = 8

#from Fred Callaway (on #psiturk channel, Feb 10 2021)
@custom_code.route('/complete_exp', methods=['POST'])
def complete_exp():
    if not 'uniqueId' in request.form:
        raise ExperimentError('improper_inputs')
    unique_id = request.form['uniqueId']
    current_app.logger.info("completed experimente")
    try:
        user = Participant.query.\
            filter(Participant.uniqueid == unique_id).one()
        user.status = COMPLETED
        user.endhit = datetime.now()
        db_session.add(user)
        db_session.commit()
        resp = {"status": "success"}
    except exc.SQLAlchemyError:
        current_app.logger.error("DB error: Unique user not found.")
        resp = {"status": "error, uniqueId not found"}
    return jsonify(**resp)

@custom_code.route('/turkprime', methods=['GET'])
@nocache
def turkprime():
    """
    This is based off the /ad router in experiment.py in psiturk
    """

    #FROM experiment.py (circular import)
    # Insert "mode" into pages so it's carried from page to page done server-side
    # to avoid breaking backwards compatibility with old templates.

    def insert_mode(page_html, mode):
        """ Insert mode """
        page_html = page_html
        match_found = False
        matches = re.finditer('workerId={{ workerid }}', page_html)
        match = None
        for match in matches:
            match_found = True
        if match_found:
            new_html = page_html[:match.end()] + "&mode=" + mode + \
                       page_html[match.end():]
            return new_html
        else:
            raise ExperimentError("insert_mode_failed")

    user_agent_string = request.user_agent.string
    user_agent_obj = user_agents.parse(user_agent_string)
    browser_ok = True

    browser_exclude_rule = config.get('HIT Configuration', 'browser_exclude_rule')
    for rule in browser_exclude_rule.split(','):
        myrule = rule.strip()
        if myrule in ["mobile", "tablet", "touchcapable", "pc", "bot"]:
            if (myrule == "mobile" and user_agent_obj.is_mobile) or\
               (myrule == "tablet" and user_agent_obj.is_tablet) or\
               (myrule == "touchcapable" and user_agent_obj.is_touch_capable) or\
               (myrule == "pc" and user_agent_obj.is_pc) or\
               (myrule == "bot" and user_agent_obj.is_bot):
                browser_ok = False
        elif myrule == "Safari" or myrule == "safari":
            if "Chrome" in user_agent_string and "Safari" in user_agent_string:
                pass
            elif "Safari" in user_agent_string:
                browser_ok = False
        elif myrule in user_agent_string:
            browser_ok = False

    if not browser_ok:
        # Handler for IE users if IE is not supported.
        raise ExperimentError('browser_type_not_allowed')

    if not ('hitId' in request.args and 'assignmentId' in request.args):
        raise ExperimentError('hit_assign_worker_id_not_set_in_mturk')
    hit_id = request.args['hitId']
    assignment_id = request.args['assignmentId']
    mode = request.args['mode']
    if hit_id[:5] == "debug":
        debug_mode = True
    else:
        debug_mode = False
    already_in_db = False
    if 'workerId' in request.args:
        worker_id = request.args['workerId']
        # First check if this workerId has completed the task before (v1).
        # Realy just checking if the worker is in the database for any other assignment
        nrecords = Participant.query.\
            filter(Participant.assignmentid != assignment_id).\
            filter(Participant.workerid == worker_id).\
            count()
        if nrecords > 0:  # Already completed task
            if 'yearLimit' in request.args:
                yearLimit = float(request.args['yearLimit'])
                times_not_allowed = date.today() - timedelta(days=yearLimit*365) #relativedelta(years=yearLimit)
                nrecords = Participant.query. \
                    filter(Participant.assignmentid != assignment_id). \
                    filter(Participant.workerid == worker_id). \
                    filter(func.date(Participant.beginhit) > times_not_allowed). \
                    count()
                if nrecords > 0:
                    already_in_db = True

            else:   #if no year limit, anyone with a databse entry for another task is excluded
                already_in_db = True
    else:  # If worker has not accepted the hit
        worker_id = None
    try:
        part = Participant.query.\
            filter(Participant.hitid == hit_id).\
            filter(Participant.assignmentid == assignment_id).\
            filter(Participant.workerid == worker_id).\
            one()
        status = part.status
    except exc.SQLAlchemyError:
        status = None

    allow_repeats = config.getboolean('HIT Configuration', 'allow_repeats')
    if (status == STARTED or status == QUITEARLY) and not debug_mode:
        # Once participants have finished the instructions, we do not allow
        # them to start the task again.
        raise ExperimentError('already_started_exp_mturk')
    elif status == COMPLETED or (status == SUBMITTED and not already_in_db):
        # 'or status == SUBMITTED' because we suspect that sometimes the post
        # to mturk fails after we've set status to SUBMITTED, so really they
        # have not successfully submitted. This gives another chance for the
        # submit to work.

        # They've finished the experiment but haven't successfully submitted the HIT
        # yet.
        return render_template(
            'complete.html'
        )
    elif already_in_db and not (debug_mode or allow_repeats):
        raise ExperimentError('already_did_exp_hit')
    elif status == ALLOCATED or not status or debug_mode:
        # Participant has not yet agreed to the consent. They might not
        # even have accepted the HIT.
        with open('templates/ad.html', 'r') as temp_file:
            ad_string = temp_file.read()
        ad_string = insert_mode(ad_string, mode)
        return render_template_string(
            ad_string,
            hitid=hit_id,
            assignmentid=assignment_id,
            workerid=worker_id
        )
    else:
        raise ExperimentError('status_incorrectly_set')

"""
Only uncomment this if you are not downloading the data in another way.
Please make login_username and login_pw (in config.txt) something relatively hard to guess.
If you keep your experiment online for a longer time period for demonstration purposes, it might be a good idea to push a version with this commented out.
"""
# def get_participants(codeversion):
#     participants = Participant\
#         .query\
#         .filter(Participant.codeversion == codeversion)\
#         .filter(Participant.status > 2)\
#         .all()
#     return participants
#
"""
Only uncomment this if you are not downloading the data in another way.
Please make login_username and login_pw (in config.txt) something relatively hard to guess.
If you keep your experiment online for a longer time period for demonstration purposes, it might be a good idea to push a version with this commented out.
"""
# @custom_code.route('/data/<codeversion>/<name>', methods=['GET'])
# @myauth.requires_auth
# @nocache
# def download_datafiles(codeversion, name):
#     contents = {
#         "trialdata": lambda p: p.get_trial_data(),
#         "eventdata": lambda p: p.get_event_data(),
#         "questiondata": lambda p: p.get_question_data()
#     }
#
#     if name not in contents:
#         abort(404)
#
#     query = get_participants(codeversion)
#     data = []
#     for p in query:
#         try:
#             data.append(contents[name](p))
#         except TypeError:
#             current_app.logger.error("Error loading {} for {}".format(name, p))
#             current_app.logger.error(format_exc())
#     ret = "".join(data)
#     response = Response(
#         ret,
#         content_type="text/csv",
#         headers={
#             'Content-Disposition': 'attachment;filename=%s.csv' % name
#         })
#
#     return response


MAX_BONUS = 10

@custom_code.route('/compute_bonus', methods=['GET'])
def compute_bonus():
    # check that user provided the correct keys
    # errors will not be that gracefull here if being
    # accessed by the Javascrip client
    if not request.args.has_key('uniqueId'):
        raise ExperimentError('improper_inputs')

    # lookup user in database
    uniqueid = request.args['uniqueId']
    user = Participant.query.\
           filter(Participant.uniqueid == uniqueid).\
           one()

    final_bonus = 'NONE'
    # load the bonus information
    try:
        all_data = json.loads(user.datastring)
        question_data = all_data['questiondata']
        final_bonus = question_data['final_bonus']
        final_bonus = round(float(final_bonus), 2)
        if final_bonus > MAX_BONUS:
            raise ValueError('Bonus of {} excedes MAX_BONUS of {}'
                             .format(final_bonus, MAX_BONUS))
        user.bonus = final_bonus
        db_session.add(user)
        db_session.commit()

        resp = {
            'uniqueId': uniqueid,
            'bonusComputed': 'success',
            'bonusAmount': final_bonus
        }

    except:
        current_app.logger.error('error processing bonus for {}'.format(uniqueid))
        current_app.logger.error(format_exc())
        resp = {
            'uniqueId': uniqueid,
            'bonusComputed': 'failure',
            'bonusAmount': final_bonus
        }

    current_app.logger.info(str(resp))
    return jsonify(**resp)
