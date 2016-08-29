#define _USE_MATH_DEFINES

#include <windows.h>
#include <iostream>

#include <GL/glut.h>  
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include <Eigen/Dense>

using namespace Eigen;

static const double LINK1_LEN = 0.3;
static const double LINK2_LEN = 0.6;
static const double STAGE_SIDE = 0.1;
static const double BASE_LEN = 0.3;
static double cam_ang_h = 1.0;
static double cam_ang_v = 1.0;
static double cam_len = 15.0;
static int mx0 = 0, my0 = 0;

VectorXd angle = VectorXd::Zero(6);
VectorXd stage_pos = VectorXd::Zero(6);
VectorXd prevPos;

Vector3d b[6] = {Vector3d(0, 0, 0)};
Vector3d s[6] = {Vector3d(0, 0, 0)};
Vector3d x1[6] = {Vector3d(0, LINK1_LEN, 0)};

double radians(double degrees){
	return M_PI/180*degrees;
}

double degrees(double radians){
	return 180/M_PI*radians;
}

Vector3d diff_base(const double link, const double base_rad, const double angle_rad){
	Vector3d d;
	d.x() =   link * sin(base_rad) * sin(angle_rad);
	d.y() = - link * cos(base_rad) * sin(angle_rad);
	d.z() =   link * cos(angle_rad);

	return d;
}

Vector3d diff_stage(const Vector3d joint_pos, const VectorXd pos, const int axis){
	assert((axis >= 0)&&(axis < 6));

	Vector3d d;
	double lx = joint_pos.x(), ly = joint_pos.y(), lz = joint_pos.z();
	double cx = cos(pos[3]), sx = sin(pos[3]),
		   cy = cos(pos[4]), sy = sin(pos[4]),
		   cz = cos(pos[5]), sz = sin(pos[5]); 
	switch(axis){
		case 0: d << 1,0,0; break;
		case 1: d << 0,1,0; break;
		case 2: d << 0,0,1; break;
		case 3: d << ly * sx - lz * cx * sz + ly * cx - lz * sx * sy,
					 ly * cx - lz * sx * sy - ly * sx - lz * cx * cz,
					 ly * cx - lz * sx * cy; break;
		case 4: d << lz * cx * cy - lx * sy * cz,
					 lz * cx * cy - lx * sy * sz,
					 - lx * cy - lz * cx * sy; break;
		case 5: d << - lz * sx * cz - lx * cy * sz,
					   lx * cy * cz + lz * sx * sz,
					 0; break;
	}
	return d;
}

void calcStageJointPos(const VectorXd stage_pos, Vector3d stage_joint_pos[6])
{
	Affine3d rot120, rot240, rot_x, rot_y, rot_z, trans;
	rot120 = AngleAxisd(radians(120), Vector3d::UnitZ());
	rot240 = AngleAxisd(radians(240), Vector3d::UnitZ());
	stage_joint_pos[0] = Vector3d( STAGE_SIDE / 2, STAGE_SIDE * sqrt(3.0) / 2, 0);
	stage_joint_pos[1] = Vector3d(-STAGE_SIDE / 2, STAGE_SIDE * sqrt(3.0) / 2, 0);
	stage_joint_pos[2] = rot120 * stage_joint_pos[0];
	stage_joint_pos[3] = rot120 * stage_joint_pos[1];
	stage_joint_pos[4] = rot240 * stage_joint_pos[0];
	stage_joint_pos[5] = rot240 * stage_joint_pos[1];
	rot_x = AngleAxisd( stage_pos[3], Vector3d::UnitX() );
	rot_y = AngleAxisd( stage_pos[4], Vector3d::UnitY() );
	rot_z = AngleAxisd( stage_pos[5], Vector3d::UnitZ() );
	trans = Translation3d(Vector3d(stage_pos.x(), stage_pos.y(), stage_pos.z()));
	for(int i = 0; i < 6; i ++){
		stage_joint_pos[i] = trans * rot_z * rot_y * rot_x * stage_joint_pos[i];
	}
}

void calcArmPos(const VectorXd joint_angle, Vector3d elbow_pos[6])
{
	for(int i = 0; i < 6; i ++){
		elbow_pos[i] << 0, LINK1_LEN, 0;
		Affine3d rot;
		rot = AngleAxisd(joint_angle[i], Vector3d::UnitX());
		elbow_pos[i] = rot * elbow_pos[i];
	}
	Affine3d rot120, rot240;
	rot120 = AngleAxisd(radians(120), Vector3d::UnitZ());
	rot240 = AngleAxisd(radians(240), Vector3d::UnitZ());
	elbow_pos[2] = rot120 * elbow_pos[2];
	elbow_pos[3] = rot120 * elbow_pos[3];
	elbow_pos[4] = rot240 * elbow_pos[4];
	elbow_pos[5] = rot240 * elbow_pos[5];
	for(int i = 0; i < 6; i ++){
		Affine3d trans;
		trans = Translation3d(b[i]);
		elbow_pos[i] = trans * elbow_pos[i];
	}
}

void calcJacoBase(const VectorXd pos, const VectorXd ang, MatrixXd& jaco_base)
{
	Vector3d elbow[6] = { Vector3d(0, LINK1_LEN, 0) };
	Vector3d stage_joint_pos[6] = {Vector3d(0, 0, 0)};
	calcStageJointPos(pos, stage_joint_pos);
	calcArmPos(ang, elbow);

	Vector3d a[6] = { Vector3d() };
	for(int i = 0;i < 6; i ++){
		a[i] = stage_joint_pos[i] - elbow[i];
	}

	jaco_base = MatrixXd::Zero(6,6);
	jaco_base(0,0) = -a[0].transpose() * diff_base(LINK1_LEN, radians(  0), ang[0]);
	jaco_base(1,1) = -a[1].transpose() * diff_base(LINK1_LEN, radians(  0), ang[1]);
	jaco_base(2,2) = -a[2].transpose() * diff_base(LINK1_LEN, radians(120), ang[2]);
	jaco_base(3,3) = -a[3].transpose() * diff_base(LINK1_LEN, radians(120), ang[3]);
	jaco_base(4,4) = -a[4].transpose() * diff_base(LINK1_LEN, radians(240), ang[4]);
	jaco_base(5,5) = -a[5].transpose() * diff_base(LINK1_LEN, radians(240), ang[5]);
//	std::cout << "matrix\n" << jaco_base << std::endl << std::endl;
}

void calcJacoStage(const VectorXd pos, const VectorXd ang, MatrixXd& jaco_stage)
{
	Vector3d elbow[6] = { Vector3d(0, LINK1_LEN, 0) };
	Vector3d stage_joint_pos[6] = {Vector3d(0, 0, 0)};
	calcStageJointPos(pos, stage_joint_pos);
	calcArmPos(ang, elbow);

	Vector3d a[6] = { Vector3d() };
	for(int i = 0;i < 6; i ++){
		a[i] = (stage_joint_pos[i] - elbow[i]);
	}

	VectorXd zero = VectorXd::Zero(6);
	Vector3d e[6] = { Vector3d(0, 0, 0) };
	calcStageJointPos(zero, e);

	jaco_stage = MatrixXd::Zero(6,6);
	for(int i = 0; i < 6; i ++){
		for(int j = 0; j < 6; j ++){
			Vector3d b = diff_stage(e[i], pos, j);
			jaco_stage(i,j) = a[i].transpose() * b;
		}
	}
//	std::cout << "matrix\n" << m << std::endl << std::endl;
}

void inv_kinematics(const VectorXd pos, VectorXd& joint_angle)
{
	static double rot[6] = {0, 0, 2.0/3.0*M_PI, 2.0/3.0*M_PI, 4.0/3.0*M_PI, 4.0/3.0*M_PI};
	Vector3d stage_joint_pos[6] = {Vector3d(0, 0, 0)};
	calcStageJointPos(pos, stage_joint_pos);
	for(int i = 0; i < 6; i ++){
		Vector3d d = stage_joint_pos[i] - b[i];
		double g = (LINK1_LEN * LINK1_LEN + d.x() * d.x() + d.y() * d.y() + d.z() * d.z() - LINK2_LEN * LINK2_LEN) / (2 * LINK1_LEN);
		double a = g + (d.y() * cos(rot[i]) - d.x() * sin(rot[i])); 
		double b = 2 * d.z();
		double c = g - (d.y() * cos(rot[i]) - d.x() * sin(rot[i]));
		double the1 = -2*atan2(-b+sqrt(b*b-4*a*c), 2*a);
		joint_angle[i] = the1;
	}
}

void kinematics(const VectorXd joint_angle, VectorXd& pos)
{
	const int max_loop = 100;
	const float limit_err = 0.001;
	VectorXd ang = VectorXd::Zero(6);
	double gain = 0.5;
	double prevErr = 1000.0;

	for(int i = 0; i < max_loop; i ++){
		inv_kinematics(pos, ang);
		VectorXd err = joint_angle - ang;
		std::cout << i << ": ERROR: " << err.norm() << std::endl;
		if (err.norm() < limit_err) break;

		MatrixXd Jb = MatrixXd::Zero(6,6), Js = MatrixXd::Zero(6,6);
		calcJacoBase(pos, ang, Jb);
		calcJacoStage(pos, ang, Js);
//		std::cout << Jb << std::endl;
		MatrixXd dpos;
		dpos = Js.inverse() * Jb * err;
		if (err.norm() > prevErr) gain /= 2;
		pos -= dpos * gain;
		prevErr = err.norm();
	}
}

void display(void)                /* CALLBACK Çè¡Ç∑ */
{
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    gluLookAt(cam_len*cos(cam_ang_h)*cos(cam_ang_v), cam_len*sin(cam_ang_h)*cos(cam_ang_v), cam_len*sin(cam_ang_v), 0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
	
	glBegin(GL_LINES);
	glNormal3f(0.0f, 0.0f, 1.0f);

	for(int i = 0, j = 1; i < 6; i ++, j = (i + 1) % 6){
		glVertex3f(b[i].x(), b[i].y(), b[i].z()); glVertex3f( b[j].x(), b[j].y(), b[j].z());
	}

	for(int i = 0; i < 6; i ++){
		glVertex3f( b[i].x(),  b[i].y(),  b[i].z()); glVertex3f( x1[i].x(), x1[i].y(), x1[i].z());
		glVertex3f(x1[i].x(), x1[i].y(), x1[i].z()); glVertex3f(  s[i].x(),  s[i].y(),  s[i].z());
	}

	for(int i = 0, j = 1; i < 6; i ++, j = (i + 1) % 6){
		glVertex3f(s[i].x(), s[i].y(), s[i].z()); glVertex3f( s[j].x(), s[j].y(), s[j].z());
	}

	glEnd();
	
    glFlush( );
}

void keyboard(unsigned char key, int x, int y)
{
	bool is_push_key = true;
	switch (key) {
		case '-': stage_pos.x() += 0.01; break;
		case 'p': stage_pos.x() -= 0.01; break;
		case 'o': stage_pos.y() += 0.01; break;
		case '@': stage_pos.y() -= 0.01; break;
		case '\\': stage_pos.z() += 0.01; break;
		case '[': stage_pos.z() -= 0.01; break;
		case ';': stage_pos[3] += 0.01; break;
		case '.': stage_pos[3] -= 0.01; break;
		case ',': stage_pos[4] += 0.01; break;
		case '/': stage_pos[4] -= 0.01; break;
		case 'l': stage_pos[5] += 0.01; break;
		case ':': stage_pos[5] -= 0.01; break;
		case ']': stage_pos << 0,0,0.7,0,0,0; break;
		default: is_push_key = false; break;
	}
	if (is_push_key){
		inv_kinematics(stage_pos, angle);
	}

	switch (key) {
		case 'q': angle[ 0] += radians(1.0f); break;
		case 'a': angle[ 0] -= radians(1.0f); break;
		case 'w': angle[ 1] += radians(1.0f); break;
		case 's': angle[ 1] -= radians(1.0f); break;
		case 'e': angle[ 2] += radians(1.0f); break;
		case 'd': angle[ 2] -= radians(1.0f); break;
		case 'r': angle[ 3] += radians(1.0f); break;
		case 'f': angle[ 3] -= radians(1.0f); break;
		case 't': angle[ 4] += radians(1.0f); break;
		case 'g': angle[ 4] -= radians(1.0f); break;
		case 'y': angle[ 5] += radians(1.0f); break;
		case 'h': angle[ 5] -= radians(1.0f); break;
		case 'z': cam_ang_h -= 0.1f; break;
		case 'x': cam_ang_h += 0.1f; break;
		case 'c': cam_ang_v -= 0.1f; break;
		case 'v': cam_ang_v += 0.1f; break;
		case 'b': cam_len -= 1.0f; break;
		case 'n': cam_len += 1.0f; break;

		case '\033': exit(0);
		default: break;
	}

	if (_finite(prevPos(1,1)) != 0) prevPos<< 0,0,0.7,0,0,0;
	kinematics(angle, prevPos);
	calcArmPos(angle, x1);
	calcStageJointPos(prevPos, s);

	glutPostRedisplay();
}

void myinit(void)
{
	Affine3d rot120, rot240;
	rot120 = AngleAxisd(radians(120), Vector3d::UnitZ());
	rot240 = AngleAxisd(radians(240), Vector3d::UnitZ());
	b[0] = Vector3d( STAGE_SIDE / 2, BASE_LEN, 0);
	b[1] = Vector3d(-STAGE_SIDE / 2, BASE_LEN, 0);
	b[2] = rot120 * b[0];
	b[3] = rot120 * b[1];
	b[4] = rot240 * b[0];
	b[5] = rot240 * b[1];

	stage_pos.z() = 0.7;
	prevPos = stage_pos;
	inv_kinematics(stage_pos, angle);
	keyboard(0, 0, 0);

	glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    glEnable(GL_DEPTH_TEST);
}

void mouse(int button , int state , int x , int y) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN){
		mx0 = x; my0 = y;
	}
}

void motion(int x , int y) {
	cam_ang_h -= (float)(x - mx0)*0.01;
	cam_ang_v += (float)(y - my0)*0.01;
	mx0 = x; my0 = y;
	glutPostRedisplay();
}

void myReshape(int w, int h) /* CALLBACK Çè¡Ç∑ */
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(10.0, (GLfloat)w/(GLfloat)h, 1.0, 10000.0);
    glMatrixMode(GL_MODELVIEW);
}

void main(void)
{
    glutInitDisplayMode (GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowPosition(0,0);
    glutInitWindowSize(500, 500);
    glutCreateWindow("test");
    myinit();
    glutReshapeFunc(myReshape);
    glutDisplayFunc(display);
	
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(keyboard);

    glutMainLoop( );
}
