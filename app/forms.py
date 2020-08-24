from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, SelectField
from wtforms.fields.html5 import IntegerField, DecimalField
from wtforms.validators import ValidationError, InputRequired, Email, EqualTo, Length, NumberRange
from app.models import User


class PatientForm(FlaskForm):
    age = IntegerField('Age', validators=[InputRequired(), NumberRange(min=1, max=150)])
    gender = SelectField('Gender', validators=[InputRequired()], choices=[(0, 'Male'), (1, 'Female')], coerce=int)
    height = IntegerField('Height (in centimeters)', validators=[InputRequired(), NumberRange(min=50, max=300)])
    weight = DecimalField('Weight (in kilograms)', validators=[InputRequired(), NumberRange(min=5, max=999)], places=1)
    ap_hi = IntegerField('Systolic blood pressure', validators=[InputRequired(), NumberRange(min=50, max=210)])
    ap_lo = IntegerField('Diastolic blood pressure', validators=[InputRequired(), NumberRange(min=20, max=120)])
    cholesterol = SelectField('Cholesterol level', validators=[InputRequired()], choices=[(1, 'Normal'), (2, 'Above normal'), (3, 'Well above normal')], coerce=int)
    gluc = SelectField('Glucose level', validators=[InputRequired()], choices=[(1, 'Normal'), (2, 'Above normal'), (3, 'Well above normal')], coerce=int)
    smoke = SelectField('Smoker', validators=[InputRequired()], choices=[(0, 'No'), (1, 'Yes')], coerce=int)
    alco = SelectField('Drinks alcohol', validators=[InputRequired()], choices=[(0, 'No'), (1, 'Yes')], coerce=int)
    active = SelectField('Physically active', validators=[InputRequired()], choices=[(0, 'No'), (1, 'Yes')], coerce=int)
    submit = SubmitField('Submit')


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    email = StringField('Email', validators=[InputRequired(), Email()])
    password = PasswordField('Password', validators=[InputRequired()])
    password2 = PasswordField('Repeat Password', validators=[InputRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')


class EditProfileForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    about_me = TextAreaField('About me', validators=[Length(min=0, max=140)])
    submit = SubmitField('Submit')

    def __init__(self, original_username, *args, **kwargs):
        super(EditProfileForm, self).__init__(*args, **kwargs)
        self.original_username = original_username

    def validate_username(self, username):
        if username.data != self.original_username:
            user = User.query.filter_by(username=self.username.data).first()
            if user is not None:
                raise ValidationError('Please use a different username.')
