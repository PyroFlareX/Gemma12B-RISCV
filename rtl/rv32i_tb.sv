`timescale 1ns/1ps
module rv32i_tb;
    localparam int IMEM_WORDS = 256;
    localparam int DMEM_WORDS = 2048;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    logic imem_valid;
    logic [31:0] imem_addr;
    logic imem_ready;
    logic [31:0] imem_rdata;
    logic imem_fault;
    logic dmem_valid;
    logic dmem_write;
    logic [31:0] dmem_addr;
    logic [31:0] dmem_wdata;
    logic [3:0] dmem_wstrb;
    logic dmem_ready;
    logic [31:0] dmem_rdata;
    logic dmem_fault;
    logic halted;
    logic trap_valid;
    logic [31:0] trap_pc;
    logic [3:0] trap_cause;
    logic [31:0] trap_value;
    logic retire_valid;
    logic [31:0] retire_pc;
    logic [31:0] retire_insn;
    logic [31:0] retire_next_pc;

    logic [31:0] instruction_memory [0:IMEM_WORDS-1];
    logic [1:0] wait_phase;
    logic [31:0] data_memory [0:DMEM_WORDS-1];

    always_ff @(posedge clk) begin
        if (!rst_n) wait_phase <= 2'b00;
        else wait_phase <= wait_phase + 1'b1;
    end
    assign imem_ready = imem_valid && wait_phase[1];
    assign imem_rdata = instruction_memory[imem_addr[9:2]];
    assign dmem_ready = dmem_valid && wait_phase[0];
    assign dmem_rdata = data_memory[dmem_addr[12:2]];

    rv32i_core #(.HALT_ON_TRAP(1'b1)) dut (
        .clk, .rst_n,
        .imem_valid, .imem_addr, .imem_ready, .imem_rdata, .imem_fault,
        .dmem_valid, .dmem_write, .dmem_addr, .dmem_wdata, .dmem_wstrb,
        .dmem_ready, .dmem_rdata, .dmem_fault,
        .halted, .trap_valid, .trap_pc, .trap_cause, .trap_value,
        .retire_valid, .retire_pc, .retire_insn, .retire_next_pc
    );

    always_ff @(posedge clk) begin
        if (rst_n && dmem_valid && dmem_ready && dmem_write && !dmem_fault) begin
            if (dmem_wstrb[0]) data_memory[dmem_addr[12:2]][7:0] <= dmem_wdata[7:0];
            if (dmem_wstrb[1]) data_memory[dmem_addr[12:2]][15:8] <= dmem_wdata[15:8];
            if (dmem_wstrb[2]) data_memory[dmem_addr[12:2]][23:16] <= dmem_wdata[23:16];
            if (dmem_wstrb[3]) data_memory[dmem_addr[12:2]][31:24] <= dmem_wdata[31:24];
        end
    end

    function automatic [31:0] enc_i(input integer imm, input integer rs1, input integer f3,
                                    input integer rd, input logic [6:0] op);
        enc_i = {imm[11:0], rs1[4:0], f3[2:0], rd[4:0], op};
    endfunction

    function automatic [31:0] enc_r(input integer f7, input integer rs2, input integer rs1,
                                    input integer f3, input integer rd, input logic [6:0] op);
        enc_r = {f7[6:0], rs2[4:0], rs1[4:0], f3[2:0], rd[4:0], op};
    endfunction

    function automatic [31:0] enc_s(input integer imm, input integer rs2, input integer rs1,
                                    input integer f3);
        enc_s = {imm[11:5], rs2[4:0], rs1[4:0], f3[2:0], imm[4:0], 7'h23};
    endfunction

    function automatic [31:0] enc_b(input integer imm, input integer rs2, input integer rs1,
                                    input integer f3);
        enc_b = {imm[12], imm[10:5], rs2[4:0], rs1[4:0], f3[2:0],
                 imm[4:1], imm[11], 7'h63};
    endfunction

    function automatic [31:0] enc_u(input integer imm20, input integer rd,
                                    input logic [6:0] op);
        enc_u = {imm20[19:0], rd[4:0], op};
    endfunction

    function automatic [31:0] enc_j(input integer imm, input integer rd);
        enc_j = {imm[20], imm[10:1], imm[11], imm[19:12], rd[4:0], 7'h6f};
    endfunction

    task automatic clear_memories;
        integer k;
        begin
            for (k = 0; k < IMEM_WORDS; k = k + 1) instruction_memory[k] = 32'h0000_0013;
            for (k = 0; k < DMEM_WORDS; k = k + 1) data_memory[k] = 32'd0;
            imem_fault = 1'b0;
            dmem_fault = 1'b0;
        end
    endtask

    task automatic reset_core;
        begin
            rst_n = 1'b0;
            repeat (2) @(posedge clk);
            @(negedge clk);
            rst_n = 1'b1;
            @(posedge clk);
            #1;
        end
    endtask

    task automatic run_to_halt(input integer max_cycles);
        integer cycles;
        begin
            cycles = 0;
            while (!halted && cycles < max_cycles) begin
                @(posedge clk);
                #1;
                cycles = cycles + 1;
            end
            if (!halted) $fatal(1, "core failed to halt within %0d cycles", max_cycles);
        end
    endtask

    task automatic check_register(input integer index, input logic [31:0] expected);
        begin
            if (dut.registers[index] !== expected)
                $fatal(1, "x%0d expected %08x got %08x", index, expected, dut.registers[index]);
        end
    endtask

    logic [127:0] q4_vec;
    logic [255:0] act_vec;
    logic dot_in_valid;
    logic dot_in_ready;
    logic dot_out_valid;
    logic dot_out_ready;
    logic signed [31:0] dot_sum;

    q4_dot_product dot_unit (
        .clk, .rst_n, .in_valid(dot_in_valid), .in_ready(dot_in_ready),
        .q4_values(q4_vec), .activations(act_vec),
        .out_valid(dot_out_valid), .out_ready(dot_out_ready), .out_sum(dot_sum)
    );

    task automatic check_dot(input logic signed [31:0] expected);
        begin
            @(negedge clk);
            if (!dot_in_ready) $fatal(1, "dot unit unexpectedly not ready");
            dot_in_valid = 1'b1;
            @(posedge clk);
            #1;
            dot_in_valid = 1'b0;
            if (!dot_out_valid || dot_sum !== expected)
                $fatal(1, "dot result expected %0d got %0d (valid=%b)", expected, dot_sum, dot_out_valid);
        end
    endtask

    integer lane;
    initial begin
        imem_fault = 1'b0;
        dmem_fault = 1'b0;
        dot_in_valid = 1'b0;
        dot_out_ready = 1'b0;
        q4_vec = '0;
        act_vec = '0;
        clear_memories();
        reset_core();

        // Scalar execution covers immediate/register arithmetic, signed and
        // unsigned loads, byte strobes, taken/not-taken branches, both jumps,
        // fence ordering, and the externally reported base ECALL exception.
        instruction_memory[0]  = enc_u(1, 1, 7'h37); // lui x1,0x1000
        instruction_memory[1]  = enc_i(-8, 0, 0, 2, 7'h13);     // addi x2, x0, -8
        instruction_memory[2]  = enc_i(3, 0, 0, 3, 7'h13);      // addi x3, x0, 3
        instruction_memory[3]  = enc_r(0, 3, 2, 0, 4, 7'h33);   // add x4, x2, x3
        instruction_memory[4]  = enc_r(7'h20, 2, 3, 0, 5, 7'h33); // sub x5, x3, x2
        instruction_memory[5]  = enc_r(0, 3, 2, 2, 6, 7'h33);   // slt x6, x2, x3
        instruction_memory[6]  = enc_r(0, 3, 2, 3, 7, 7'h33);   // sltu x7, x2, x3
        instruction_memory[7]  = enc_i(2, 3, 1, 8, 7'h13);      // slli x8, x3, 2
        instruction_memory[8]  = enc_i((7'h20 << 5) | 1, 2, 5, 9, 7'h13); // srai x9,x2,1
        instruction_memory[9]  = enc_i(16'h055, 0, 6, 10, 7'h13); // ori x10,x0,0x55
        instruction_memory[10] = enc_s(0, 10, 1, 2);            // sw x10,0(x1)
        instruction_memory[11] = enc_s(1, 2, 1, 0);             // sb x2,1(x1)
        instruction_memory[12] = enc_i(1, 1, 4, 11, 7'h03);    // lbu x11,1(x1)
        instruction_memory[13] = enc_i(1, 1, 0, 12, 7'h03);    // lb x12,1(x1)
        instruction_memory[14] = enc_i(0, 1, 1, 13, 7'h03);    // lh x13,0(x1)
        instruction_memory[15] = enc_s(4, 11, 1, 2);            // sw x11,4(x1)
        instruction_memory[16] = enc_i(4, 1, 2, 14, 7'h03);    // lw x14,4(x1)
        instruction_memory[17] = enc_b(8, 14, 11, 0);          // beq x11,x14,+8
        instruction_memory[18] = enc_i(99, 0, 0, 15, 7'h13);   // skipped
        instruction_memory[19] = enc_b(8, 14, 11, 1);          // bne not taken
        instruction_memory[20] = enc_b(8, 11, 14, 5);          // bge taken
        instruction_memory[21] = enc_i(77, 0, 0, 15, 7'h13);   // skipped
        instruction_memory[22] = enc_j(8, 16);                 // jal x16,+8
        instruction_memory[23] = enc_i(99, 0, 0, 17, 7'h13);   // skipped
        instruction_memory[24] = enc_u(0, 18, 7'h17);           // auipc x18,0
        instruction_memory[25] = enc_i(112, 0, 0, 20, 7'h13);  // target for jalr
        instruction_memory[26] = enc_i(0, 20, 0, 21, 7'h67);   // jalr x21,0(x20)
        instruction_memory[27] = enc_i(99, 0, 0, 17, 7'h13);   // skipped
        instruction_memory[28] = enc_i(17, 0, 0, 22, 7'h13);  // addi x22,x0,17
        instruction_memory[29] = 32'h0000_000f;                // fence
        instruction_memory[30] = 32'h0000_0073;                // ecall
        run_to_halt(100);

        if (!trap_valid || trap_cause != 4'd11 || trap_pc != 32'd120 || trap_value != 0)
            $fatal(1, "ECALL trap metadata incorrect pc=%08x cause=%0d value=%08x",
                   trap_pc, trap_cause, trap_value);
        check_register(4, 32'hffff_fffb);
        check_register(5, 32'd11);
        check_register(6, 32'd1);
        check_register(7, 32'd0);
        check_register(8, 32'd12);
        check_register(9, 32'hffff_fffc);
        check_register(11, 32'd248);
        check_register(12, 32'hffff_fff8);
        check_register(13, 32'hffff_f855);
        check_register(14, 32'd248);
        check_register(15, 32'd0);
        check_register(16, 32'd92);
        check_register(17, 32'd0);
        check_register(18, 32'd96);
        check_register(21, 32'd108);
        check_register(22, 32'd17);
        if (data_memory[1024] !== 32'h0000_f855 || data_memory[1025] !== 32'd248)
            $fatal(1, "byte/word stores incorrect: %08x %08x", data_memory[1024], data_memory[1025]);
        if (retire_valid) $fatal(1, "ECALL must trap, not retire");

        // Exercise the remaining immediate/register ALU operations and signed
        // versus unsigned comparisons/shifts in a second instruction stream.
        clear_memories();
        instruction_memory[0]  = enc_i(-1, 0, 0, 1, 7'h13); // addi x1,x0,-1
        instruction_memory[1]  = enc_i(1, 0, 0, 2, 7'h13);  // addi x2,x0,1
        instruction_memory[2]  = enc_i(0, 1, 2, 3, 7'h13);  // slti x3,x1,0
        instruction_memory[3]  = enc_i(1, 1, 3, 4, 7'h13);  // sltiu x4,x1,1
        instruction_memory[4]  = enc_i('h55, 1, 4, 5, 7'h13); // xori
        instruction_memory[5]  = enc_i('h5a, 1, 7, 6, 7'h13); // andi
        instruction_memory[6]  = enc_i('h80, 2, 6, 7, 7'h13); // ori
        instruction_memory[7]  = enc_i(31, 2, 1, 8, 7'h13);  // slli
        instruction_memory[8]  = enc_i(31, 8, 5, 9, 7'h13);  // srli
        instruction_memory[9]  = enc_i((7'h20 << 5) | 4, 1, 5, 10, 7'h13); // srai
        instruction_memory[10] = enc_r(0, 2, 2, 1, 11, 7'h33); // sll
        instruction_memory[11] = enc_r(0, 2, 8, 5, 12, 7'h33); // srl
        instruction_memory[12] = enc_r(7'h20, 2, 1, 5, 13, 7'h33); // sra
        instruction_memory[13] = enc_r(0, 2, 1, 2, 14, 7'h33); // slt
        instruction_memory[14] = enc_r(0, 2, 1, 3, 15, 7'h33); // sltu
        instruction_memory[15] = enc_r(0, 2, 1, 4, 16, 7'h33); // xor
        instruction_memory[16] = enc_r(0, 2, 1, 6, 17, 7'h33); // or
        instruction_memory[17] = enc_r(0, 2, 1, 7, 18, 7'h33); // and
        instruction_memory[18] = enc_r(7'h20, 1, 2, 0, 19, 7'h33); // sub
        instruction_memory[19] = enc_b(8, 2, 1, 4); // blt taken
        instruction_memory[20] = enc_i(99, 0, 0, 30, 7'h13); // skipped
        instruction_memory[21] = enc_b(8, 1, 2, 5); // bge taken
        instruction_memory[22] = enc_i(99, 0, 0, 30, 7'h13); // skipped
        instruction_memory[23] = enc_b(8, 2, 1, 6); // bltu not taken
        instruction_memory[24] = enc_b(8, 2, 1, 7); // bgeu x1,x2 taken (0xffffffff >= 1)
        instruction_memory[25] = enc_i(99, 0, 0, 30, 7'h13); // skipped
        instruction_memory[26] = enc_b(8, 1, 2, 1); // bne taken
        instruction_memory[27] = enc_i(99, 0, 0, 30, 7'h13); // skipped
        instruction_memory[28] = enc_b(8, 2, 1, 0); // beq not taken
        instruction_memory[29] = 32'h0010_0073; // ebreak
        reset_core();
        run_to_halt(100);
        if (trap_cause != 4'd3 || trap_pc != 116 || trap_value != 116)
            $fatal(1, "EBREAK trap metadata incorrect");
        check_register(3, 1);
        check_register(4, 0);
        check_register(5, 32'hffff_ffaa);
        check_register(6, 32'h5a);
        check_register(7, 32'h81);
        check_register(8, 32'h8000_0000);
        check_register(9, 1);
        check_register(10, 32'hffff_ffff);
        check_register(11, 2);
        check_register(12, 32'h4000_0000);
        check_register(13, 32'hffff_ffff);
        check_register(14, 1);
        check_register(15, 0);
        check_register(16, 32'hffff_fffe);
        check_register(17, 32'hffff_ffff);
        check_register(18, 1);
        check_register(19, 2);
        check_register(30, 0);

        // Misaligned scalar accesses are handled in hardware with two aligned word accesses.
        clear_memories();
        data_memory[1] = 32'h0403_0201;
        data_memory[2] = 32'h0605_0403;
        instruction_memory[0] = enc_i(4, 0, 0, 1, 7'h13);      // x1 = 4
        instruction_memory[1] = enc_i(3, 1, 2, 2, 7'h03);      // lw x2, 3(x1) -> addr 7, crosses words
        instruction_memory[2] = enc_i(1, 1, 1, 3, 7'h03);      // lh x3, 1(x1) -> addr 5
        instruction_memory[3] = 32'h0000_0073;                // ecall -> halt (HALT_ON_TRAP)
        reset_core();
        run_to_halt(20);
        if (dut.registers[2] !== 32'h0504_0304)
            $fatal(1, "misaligned lw read %08x, expected 05040304", dut.registers[2]);
        if (dut.registers[3] !== 32'h0000_0302)
            $fatal(1, "misaligned lh read %08x, expected 00000302", dut.registers[3]);

        clear_memories();
        instruction_memory[0] = enc_i(4, 0, 0, 1, 7'h13);      // x1 = 4
        instruction_memory[1] = enc_u(32'h12345, 2, 7'h37);    // lui x2, 0x12345 -> 0x12345000
        instruction_memory[2] = enc_s(1, 2, 1, 3'b010);        // sw x2, 1(x1) -> addr 5, spans two words
        instruction_memory[3] = 32'h0000_0073;                // ecall -> halt
        reset_core();
        run_to_halt(20);
        if (data_memory[1] !== 32'h3450_0000 || data_memory[2] !== 32'h0000_0012)
            $fatal(1, "misaligned sw wrote %08x/%08x, expected 34500000/00000012",
                   data_memory[1], data_memory[2]);

        clear_memories();
        instruction_memory[0] = enc_i(0, 0, 2, 1, 7'h03); // lw x1,0(x0)
        dmem_fault = 1'b1;
        reset_core();
        run_to_halt(8);
        if (trap_cause != 4'd5 || trap_pc != 0 || trap_value != 0)
            $fatal(1, "load-access trap metadata incorrect");

        clear_memories();
        instruction_memory[0] = enc_s(0, 0, 0, 2); // sw x0,0(x0)
        dmem_fault = 1'b1;
        reset_core();
        run_to_halt(8);
        if (trap_cause != 4'd7 || trap_pc != 0 || trap_value != 0)
            $fatal(1, "store-access trap metadata incorrect");

        // Independent custom Q4xINT8 lane-parallel dot datapath checks signed
        // products and elastic output backpressure/valid retention.
        clear_memories();
        dmem_fault = 1'b0;
        reset_core();
        for (lane = 0; lane < 32; lane = lane + 1) begin
            q4_vec[lane*4 +: 4] = 4'h1;
            act_vec[lane*8 +: 8] = 8'h02;
        end
        dot_out_ready = 1'b0;
        check_dot(32'sd64);
        if (!dot_out_valid || dot_sum != 64) $fatal(1, "dot result did not hold under backpressure");
        @(negedge clk);
        dot_out_ready = 1'b1;
        @(posedge clk);
        #1;
        if (dot_out_valid) $fatal(1, "dot valid failed to clear after output handshake");
        @(negedge clk);
        dot_out_ready = 1'b0;
        for (lane = 0; lane < 32; lane = lane + 1) begin
            q4_vec[lane*4 +: 4] = 4'h8; // -8
            act_vec[lane*8 +: 8] = 8'h80; // -128
        end
        check_dot(32'sd32768);
        if (!dot_out_valid || dot_sum != 32768) $fatal(1, "signed dot result invalid");
        @(negedge clk);
        dot_out_ready = 1'b1;
        @(posedge clk);
        #1;
        if (dot_out_valid) $fatal(1, "dot valid failed to clear after second handshake");

        $display("rv32i_tb PASS");
        $finish;
    end
endmodule
