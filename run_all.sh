scripts=(
    "illness.sh"
    "weather.sh"
    "electricity.sh"
    "traffic.sh"
    "etth2.sh"
    "ettm1.sh"
    "ettm2.sh"
    "etth1.sh"
    "exchange.sh"
)

for script in ${scripts[@]};
do
    bash "scripts/rec_enc/${script}"
done