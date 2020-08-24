from datetime import datetime
from app import db, login
from sqlalchemy.dialects.postgresql import UUID
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from hashlib import md5


followers = db.Table('followers',
                     db.Column('follower_id', db.Integer, db.ForeignKey('user.id')),
                     db.Column('followed_id', db.Integer, db.ForeignKey('user.id'))
                     )


@login.user_loader
def load_user(id):
    return User.query.get(int(id))


class CremeModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), index=True, unique=True)
    pipeline = db.Column(db.PickleType)

    def __repr__(self):
        return '<Creme model {}>'.format(self.name)

    def fit_one(self, x, y):
        self.pipeline.fit_one(x, y)
        return self

    def predict_one(self, x):
        return self.pipeline.predict_one(x)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    rating = db.Column(db.Integer)
    reputation = db.Column(db.Integer)
    activeness = db.Column(db.Integer)
    num_of_feedback_given = db.Column(db.Integer)
    good_doctor = db.Column(db.Boolean)
    patients = db.relationship('Patient', backref='doctor', lazy='dynamic')
    about_me = db.Column(db.String(140))
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    followed = db.relationship(
        'User', secondary=followers,
        primaryjoin=(followers.c.follower_id == id),
        secondaryjoin=(followers.c.followed_id == id),
        backref=db.backref('followers', lazy='dynamic'), lazy='dynamic')

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def avatar(self, size):
        digest = md5(self.email.lower().encode('utf-8')).hexdigest()
        return 'https://www.gravatar.com/avatar/{}?d=identicon&s={}'.format(digest, size)

    def follow(self, user):
        if not self.is_following(user):
            self.followed.append(user)

    def unfollow(self, user):
        if self.is_following(user):
            self.followed.remove(user)

    def is_following(self, user):
        return self.followed.filter(followers.c.followed_id == user.id).count() > 0

    def followed_patients(self):
        followed = Patient.query.join(
            followers, (followers.c.followed_id == Patient.user_id)).filter(
                followers.c.follower_id == self.id)
        own = Patient.query.filter_by(user_id=self.id)
        return followed.union(own).order_by(Patient.timestamp.desc())


class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.Integer)
    height = db.Column(db.Integer)
    weight = db.Column(db.Float)
    ap_hi = db.Column(db.Integer)
    ap_lo = db.Column(db.Integer)
    cholesterol = db.Column(db.Integer)
    gluc = db.Column(db.Integer)
    smoke = db.Column(db.Integer)
    alco = db.Column(db.Integer)
    active = db.Column(db.Integer)
    bmi = db.Column(db.Float)
    prediction = db.Column(db.Float)
    feedback = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    def __repr__(self):
        return '<Patient {}: Age: {}, Gender: {}, Height: {}, Weight: {}>'\
            .format(self.id, self.age, self.gender, self.height, self.weight)
