SCRIPTS := $(CURDIR)/src/*.py
LOCAL_BIN := $(HOME)/.local/bin

all: backup

migrate:
	mv -u $(SCRIPTS) $(LOCAL_BIN)
	

