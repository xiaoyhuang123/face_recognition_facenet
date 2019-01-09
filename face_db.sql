/*
Navicat MySQL Data Transfer

Source Server         : hhydemodatabase
Source Server Version : 50527
Source Host           : 127.0.0.1:3306
Source Database       : face_db

Target Server Type    : MYSQL
Target Server Version : 50527
File Encoding         : 65001

Date: 2019-01-09 14:14:49
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for face_data
-- ----------------------------
DROP TABLE IF EXISTS `face_data`;
CREATE TABLE `face_data` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `f_name` varchar(100) DEFAULT NULL,
  `f_encode` varchar(5000) DEFAULT NULL,
  `f_file_name` varchar(500) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=428 DEFAULT CHARSET=utf8;
