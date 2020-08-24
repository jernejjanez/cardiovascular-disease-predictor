from flask import render_template, flash, redirect, url_for, request, jsonify
from flask_login import current_user, login_user, logout_user, login_required
from app import app, db
from app.ml_model import diagnose, calculate_bmi
from app.feedback import feedback
from app.forms import LoginForm, RegistrationForm, EditProfileForm, PatientForm
from app.models import User, Patient, CremeModel
from werkzeug.urls import url_parse
from datetime import datetime
from random import randint
import pickle


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    form = PatientForm()
    if form.validate_on_submit():
        patient = Patient(age=form.age.data,
                          gender=form.gender.data,
                          height=form.height.data,
                          weight=form.weight.data,
                          ap_hi=form.ap_hi.data,
                          ap_lo=form.ap_lo.data,
                          cholesterol=form.cholesterol.data,
                          gluc=form.gluc.data,
                          smoke=form.smoke.data,
                          alco=form.alco.data,
                          active=form.active.data,
                          doctor=current_user)
        patient.bmi = calculate_bmi(patient.weight, patient.height)
        patient.prediction = diagnose(patient)
        db.session.add(patient)
        db.session.commit()
        flash('You added a new patient!')
        return redirect(url_for('index'))
    page = request.args.get('page', 1, type=int)
    patients = current_user.patients.order_by(Patient.timestamp.desc()).paginate(page, app.config['PATIENTS_PER_PAGE'], False)
    next_url = url_for('index', page=patients.next_num) if patients.has_next else None
    prev_url = url_for('index', page=patients.prev_num) if patients.has_prev else None
    return render_template('index.html', title="Home", form=form, patients=patients.items, next_url=next_url, prev_url=prev_url,
                           show_feedback_modal=True)


@app.route('/diagnose', methods=['POST'])
@login_required
def diagnose_patient():
    patient = Patient.query.get(request.form['patient_id'])
    patient.prediction = diagnose(patient)
    db.session.commit()
    return jsonify({'prediction': patient.prediction})


@app.route('/feedback', methods=['POST'])
@login_required
def give_feedback():
    patient = Patient.query.get(request.form['patient_id'])
    patient.feedback = request.form['feedback_value']
    current_user.num_of_feedback_given += 1
    if current_user.num_of_feedback_given == app.config['NUM_OF_FEEDBACK_NEEDED_FOR_EVALUATION']:
        model = CremeModel.query.filter_by(name="BestModel").first()
    db.session.commit()
    return jsonify({'feedback': patient.feedback})


@app.route('/explore')
@login_required
def explore():
    page = request.args.get('page', 1, type=int)
    patients = Patient.query.order_by(Patient.timestamp.desc()).paginate(page, app.config['PATIENTS_PER_PAGE'], False)
    next_url = url_for('explore', page=patients.next_num) if patients.has_next else None
    prev_url = url_for('explore', page=patients.prev_num) if patients.has_prev else None
    return render_template('index.html', title='Explore', patients=patients.items, next_url=next_url, prev_url=prev_url)


@app.route('/user/<username>')
@login_required
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    page = request.args.get('page', 1, type=int)
    patients = user.patients.order_by(Patient.timestamp.desc()).paginate(page, app.config['PATIENTS_PER_PAGE'], False)
    next_url = url_for('user', username=user.username, page=patients.next_num) if patients.has_next else None
    prev_url = url_for('user', username=user.username, page=patients.prev_num) if patients.has_prev else None
    return render_template('user.html', title='Profile', user=user, patients=patients.items, next_url=next_url, prev_url=prev_url)


@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm(current_user.username)
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash('Your changes have been saved.')
        return redirect(url_for('edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', title='Edit Profile', form=form)


@app.route('/follow/<username>')
@login_required
def follow(username):
    user = User.query.filter_by(username=username).first()
    if user is None:
        flash('User {} not found.'.format(username))
        return redirect(url_for('index'))
    if user == current_user:
        flash('You cannot follow yourself!')
        return redirect(url_for('user', username=username))
    current_user.follow(user)
    db.session.commit()
    flash('You are following {}!'.format(username))
    return redirect(url_for('user', username=username))


@app.route('/unfollow/<username>')
@login_required
def unfollow(username):
    user = User.query.filter_by(username=username).first()
    if user is None:
        flash('User {} not found.'.format(username))
        return redirect(url_for('index'))
    if user == current_user:
        flash('You cannot unfollow yourself!')
        return redirect(url_for('user', username=username))
    current_user.unfollow(user)
    db.session.commit()
    flash('You are not following {} anymore.'.format(username))
    return redirect(url_for('user', username=username))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        user.good_doctor = False
        user.num_of_feedback_given = 0
        user.rating = randint(1, 5)
        user.reputation = randint(1, 5)
        user.activeness = randint(1, 5)
        if CremeModel.query.filter_by(name="BestModel").first() is None:
            model = CremeModel(name="BestModel", pipeline=pickle.load(open("online_learning/initial_creme_model.sav", "rb")))
            db.session.add(model)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()
