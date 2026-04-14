terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "ap-northeast-1"
}

resource "aws_ecr_repository" "ops_copilot_ecr" {
  name = "ops-copilot-tf"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_security_group" "ops_copilot_sg" {
  name = "ops-copilot-sg"

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}