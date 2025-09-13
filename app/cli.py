import click
from .extensions import db
from .models import User

def init_cli(app):
    @app.cli.command("create-user")
    @click.option("--email", prompt=True)
    @click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
    @click.option("--admin", is_flag=True, default=False)
    def create_user(email, password, admin):
        email = email.strip().lower()
        if User.query.filter_by(email=email).first():
            click.echo("User already exists"); return
        u = User(email=email, role="admin" if admin else "user")
        u.set_password(password)
        db.session.add(u)
        db.session.commit()
        click.echo(f"Created user {email} (role={u.role})")
