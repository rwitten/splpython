#! /bin/csh -f

if("$1" == "") then
    echo "Usage: executeAndLog.sh <LOG> <RUN> <CMD>";
    exit
endif

set log = $1; shift;
set RUN = $1; shift;

if($log == "/dev/null") then
    set tmpLog = "/dev/null";
else if(-e $log || -e $log.gz) then
    exit
else if($RUN) then
    set ID = "`hostname` $$";
    echo $ID >>! $log
    set tmpLog = "/tmp/$log:t.pid$$"
    echo $ID >>! $tmpLog
    set line = `head -1 $log`
    if ( "$line" != "$ID" ) then
        exit
    endif
endif

set cmd = "";
while ("$1" != "")
    set cmd = "$cmd $1"; shift;
end

echo @ $cmd
if($RUN) then
    set start = `date`;

    # LOG THE COMMAND
    echo " @ " `hostname` "|" $start "|" $cmd "|" $log >> command.log

    $cmd >>& $tmpLog;
    set finish = `date`;
    echo $cmd >> $tmpLog;
    echo $start >> $tmpLog;
    echo $finish >> $tmpLog;

    if ( $log != "/dev/null" ) then
	gzip -f $tmpLog
	mv $tmpLog.gz $log.gz
	rm $log
    endif

endif


